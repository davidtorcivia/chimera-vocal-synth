//! Debug: render PIE text and print the airway's narrowest point and noise sources at given times.
//!   piedebug "text" male t1 t2 ... [key=value PieOptions]
use chimera::anatomy::human::{BODY_FEMALE, BODY_MALE};
use chimera::phonetics::pie::{PieOptions, plan_pie, setup_voice};
use chimera::phonetics::planner::PlanOptions;
use chimera::voice::human_voice::HumanVoice;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let text = &a[1];
    let body_name = a[2].as_str();
    let mut times: Vec<f64> = Vec::new();
    let mut opts = serde_json::to_value(PieOptions::default()).unwrap();
    for x in &a[3..] {
        if let Some((k, v)) = x.split_once('=') {
            opts[k] = serde_json::from_str(v).unwrap();
        } else {
            times.push(x.parse().unwrap());
        }
    }
    let p: PieOptions = serde_json::from_value(opts).unwrap();
    let (body, f0) = if body_name == "female" { (BODY_FEMALE, 200.0) } else { (BODY_MALE, 110.0) };
    chimera::init();
    let mut v = HumanVoice::new(48000.0, body, 2);
    setup_voice(&mut v, f0, &p);
    let plan = plan_pie(text, &PlanOptions { body: body_name.into(), t0: 0.05, f0: Some(f0), ..Default::default() }, &p);
    for s in &plan.segments {
        print!("{}@{:.3} ", s.ipa, s.t0);
    }
    println!();
    v.schedule(&plan.gestures);
    let mut buf = vec![0f32; 128];
    let mut t = 0.0;
    let mut ti = 0;
    while ti < times.len() {
        v.render(&mut buf);
        t += 128.0 / 48000.0;
        if t >= times[ti] {
            let air = &v.airway;
            let all: Vec<f64> = air.ph_min.iter().chain(air.or_min.iter()).cloned().collect();
            let (mi, ma) = all.iter().enumerate().filter(|(_, a)| **a > 0.0).fold((0, f64::INFINITY), |m, (i, a)| if *a < m.1 { (i, *a) } else { m });
            println!("t={:.3} narrowest #{mi}/{} (pharynx {}) area {ma:.3} cm²  glottis area {:.3}", t, all.len(), air.n_ph, v.glottis.area);
            for s in air.net.noise_sources() {
                println!("   noise src {} gain {:.1e} level {:.3e}", s.src, s.gain, s.level);
            }
            ti += 1;
        }
    }
}
