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
    if let Ok(x) = std::env::var("NG_GLOTTIS") {
        v.airway.noise_gain.glottis *= x.parse::<f64>().unwrap();
    }
    if let Ok(x) = std::env::var("NG_CONSTR") {
        v.airway.noise_gain.constriction *= x.parse::<f64>().unwrap();
    }
    let mut whole: Vec<f32> = Vec::new();
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
        whole.extend_from_slice(&buf);
        t += 128.0 / 48000.0;
        if t >= times[ti] {
            let air = &v.airway;
            let all: Vec<f64> = air.ph_min.iter().chain(air.or_min.iter()).cloned().collect();
            let (mi, ma) = all.iter().enumerate().filter(|(_, a)| **a > 0.0).fold((0, f64::INFINITY), |m, (i, a)| if *a < m.1 { (i, *a) } else { m });
            println!("t={:.3} narrowest #{mi}/{} (pharynx {}) area {ma:.3} cm²  glottis area {:.3}  velum port {:.3}  voicing {:.2}  dp {:.1} cmH2O  sub {:.1} supra {:.1}", t, all.len(), air.n_ph, v.glottis.area, air.frame.as_ref().map_or(f64::NAN, |f| f.velum_port), v.glottis.voicing, v.glottis.dp_sm / 980.665, air.net.p[air.trachea.last_node()] / 980.665, air.net.p[air.pharynx.first_node()] / 980.665);
            if std::env::var("AREAS").is_ok() {
                if let Some(f) = &air.frame {
                    println!("   length {:.2} pharynx {:.2} teeth_s {:.2} lip_s {:.2}", f.length, f.pharynx_length, f.teeth_s, f.lip_s);
                    let dx_or = air.or_len / air.or_min.len() as f64;
                    let row: Vec<String> = air.or_min.iter().enumerate().map(|(i, a)| format!("{:.2}@{:.1}", a, air.ph_len + (i as f64 + 0.5) * dx_or)).collect();
                    println!("   oral (area@s): {}", row.join(" "));
                }
            }
            for s in air.net.noise_sources() {
                println!("   noise src {} gain {:.1e} level {:.3e} Re {:.0}", s.src, s.gain, s.level, s.re);
            }
            ti += 1;
        }
    }
    if let Ok(path) = std::env::var("WAV") {
        // keep rendering to the end of the plan, then write 32-bit float
        while t < plan.end + 0.3 {
            v.render(&mut buf);
            whole.extend_from_slice(&buf);
            t += 128.0 / 48000.0;
        }
        let n = whole.len() as u32;
        let mut w = Vec::new();
        w.extend_from_slice(b"RIFF");
        w.extend_from_slice(&(36 + n * 4).to_le_bytes());
        w.extend_from_slice(b"WAVEfmt ");
        for x in [16u32] { w.extend_from_slice(&x.to_le_bytes()); }
        w.extend_from_slice(&3u16.to_le_bytes());
        w.extend_from_slice(&1u16.to_le_bytes());
        w.extend_from_slice(&48000u32.to_le_bytes());
        w.extend_from_slice(&(48000u32 * 4).to_le_bytes());
        w.extend_from_slice(&4u16.to_le_bytes());
        w.extend_from_slice(&32u16.to_le_bytes());
        w.extend_from_slice(b"data");
        w.extend_from_slice(&(n * 4).to_le_bytes());
        for x in &whole { w.extend_from_slice(&x.to_le_bytes()); }
        std::fs::write(path, w).unwrap();
    }
}
