//! Proto-Indo-European through the engine (phonetics::pie): the sounds it has to make.

use chimera::anatomy::human::BODY_MALE;
use chimera::phonetics::pie::{PieOptions, plan_pie, setup_voice};
use chimera::phonetics::planner::PlanOptions;
use chimera::voice::human_voice::HumanVoice;

/// Render `text` and call `at(voice, t)` after each block.
fn run(text: &str, p: &PieOptions, mut at: impl FnMut(&HumanVoice, f64)) -> Vec<f32> {
    chimera::init();
    let mut v = HumanVoice::new(48000.0, BODY_MALE, 2);
    setup_voice(&mut v, 110.0, p);
    let plan = plan_pie(text, &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(110.0), ..Default::default() }, p);
    v.schedule(&plan.gestures);
    let n = ((plan.end + 0.3) * 48000.0) as usize;
    let mut out = vec![0f32; n];
    let mut t = 0.0;
    for b in out.chunks_mut(128) {
        v.render(b);
        t += b.len() as f64 / 48000.0;
        at(&v, t);
    }
    out
}

/// A uvular constriction sits just past the pharynx/mouth junction. Its frication source used to
/// be dropped there (the upstream position landed on the junction, which has no branch), so [χ]
/// was only breath from the glottis, an [h].
#[test]
fn a_uvular_fricative_has_its_own_frication() {
    let plan = plan_pie("áh₂a", &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(110.0), ..Default::default() }, &PieOptions::default());
    let chi = plan.segments.iter().find(|s| s.ipa == "χ").unwrap();
    let mid = 0.5 * (chi.t0 + chi.t1);
    let mut sources_mid = 0;
    let mut oral_level = 0.0f64;
    run("áh₂a", &PieOptions::default(), |v, t| {
        if (t - mid).abs() < 0.0014 {
            let ns = v.airway.net.noise_sources();
            sources_mid = ns.len();
            oral_level = ns.iter().skip(1).map(|s| s.level).fold(0.0, f64::max);
        }
    });
    assert!(sources_mid >= 2, "only {sources_mid} noise source(s) in the middle of χ");
    assert!(oral_level > 0.0, "the oral source is silent");
}

/// Murmur keeps the folds vibrating: the vowel after dʰ is voiced from its start, as after d.
#[test]
fn murmured_release_stays_voiced() {
    let mut voicing = Vec::new();
    let plan = plan_pie("ádʰa", &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(110.0), ..Default::default() }, &PieOptions::default());
    let a2 = plan.segments.last().unwrap().t0;
    run("ádʰa", &PieOptions::default(), |v, t| {
        if t > a2 + 0.005 && t < a2 + 0.04 {
            voicing.push(v.glottis.voicing);
        }
    });
    let mean = voicing.iter().sum::<f64>() / voicing.len() as f64;
    assert!(mean > 0.5, "voicing after the murmured release {mean:.2}");
}

/// The pitch accent: the accented vowel's planned pitch stands above the unaccented ones around it,
/// by about the model's accent range, and the line declines.
#[test]
fn the_accent_rises_and_the_line_declines() {
    use chimera::control::score::ControlKey;
    let o = PlanOptions { body: "male".into(), t0: 0.05, f0: Some(110.0), ..Default::default() };
    let p = PieOptions::default();
    let plan = plan_pie("ph₂tḗr bʰréh₂tēr", &o, &p);
    let hz_at = |t: f64| {
        plan.gestures
            .iter()
            .filter(|g| g.label_str() == "pitch" && g.t0 <= t && t < g.t1)
            .filter_map(|g| g.targets.get(ControlKey::F0))
            .next()
            .unwrap()
    };
    let st = |hz: f64| 12.0 * (hz / 110.0).log2();
    let nuc: Vec<_> = plan.segments.iter().filter(|s| s.ipa.starts_with(['a', 'e', 'o'])).collect();
    let acc = nuc.iter().find(|s| s.ipa.contains('\u{301}')).unwrap();
    let un = nuc.iter().find(|s| !s.ipa.contains('\u{301}')).unwrap();
    let lift = st(hz_at(0.5 * (acc.t0 + acc.t1))) - st(hz_at(0.5 * (un.t0 + un.t1)));
    assert!(lift > 0.6 * p.accent_st, "accent lift {lift:.2} st");
    let first = plan.segments.first().unwrap();
    let last = plan.segments.last().unwrap();
    assert!(st(hz_at(first.t0)) > st(hz_at(last.t1 - 0.01)), "the line should fall");
}
