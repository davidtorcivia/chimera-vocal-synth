//! PIE through CHIMERA: utterances with exact phone durations (from our VTL renders) plus the PIE
//! features its IPA reader drops. `cargo run --release -p chimera-cli --example pie -- in.json out.wav male|female`
//!   ʷ  lips rounded (as for /u/) through the stop and 30 ms into the next sound
//!   ʱ  murmur: the folds part slightly at the release, 40 ms into the vowel, pitch 2 st lower
//!   voiceless stops unaspirated (the planner aspirates them English-style)
//!   accented syllabic resonants get the accent's pitch
use chimera::anatomy::human::{ArticKey, BODY_FEMALE, BODY_MALE};
use chimera::control::score::{ControlKey, Gesture, Targets};
use chimera::phonetics::Segment;
use chimera::phonetics::planner::{PlanOptions, plan_utterance};
use chimera::voice::human_voice::{HumanVoice, SourcePatch};
use serde_json::Value;

const FS: f64 = 48000.0;
const LEAD: f64 = 0.06; // the planner's breath-onset lead before the first segment

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let src: Value = serde_json::from_str(&std::fs::read_to_string(&a[1]).unwrap()).unwrap();
    let out = &a[2];
    let body_name = a.get(3).map_or("male", |s| s.as_str());
    let accent: f64 = a.get(4).map_or(0.2, |s| s.parse().unwrap());
    let (body, f0) = if body_name == "female" { (BODY_FEMALE, 200.0) } else { (BODY_MALE, 110.0) };
    chimera::init();
    let mut voice = HumanVoice::new(FS, body, 2);
    voice.set_source(&SourcePatch::f0(f0));
    let mut end = 0.0f64;
    for u in src["utterances"].as_array().unwrap() {
        let segs_j = u["segs"].as_array().unwrap();
        let start = u["start"].as_f64().unwrap() + 0.3; // 0.3 s of room for the first breath
        let segs: Vec<Segment> = segs_j
            .iter()
            .map(|s| Segment {
                ipa: s["ipa"].as_str().unwrap().to_string(),
                dur: s["dur"].as_f64(),
                stress: Some(s["stress"].as_u64().unwrap() as u8),
                f0: None,
                word_end: Some(s["word_end"].as_bool().unwrap()),
            })
            .collect();
        let mut plan = plan_utterance(&segs, &PlanOptions { body: body_name.into(), t0: start - LEAD, f0: Some(f0), accent: Some(accent), ..Default::default() });
        // PIE voiceless stops are plain: the English aspiration gesture becomes the unaspirated one
        for g in plan.gestures.iter_mut() {
            if g.label_str() == "aspiration" {
                g.targets = Targets::default().with(ControlKey::ABDUCTION, 0.22);
                g.t1 -= 0.015;
                g.ramp_out = 0.02;
            }
        }
        let ps = plan.segments.clone();
        for (i, s) in segs_j.iter().enumerate() {
            let (t0, t1) = (ps[i].t0, ps[i].t1);
            if s["labial"].as_bool().unwrap() {
                let tg = Targets::default().with(ArticKey::LipAp, 0.3).with(ArticKey::LipProt, 1.4).with(ArticKey::LipWidth, 0.065);
                plan.gestures.push(Gesture::new(t0 - 0.02, t1 + 0.03, 0.03, 0.03, 3.0, tg).label("labial"));
            }
            if s["breathy"].as_bool().unwrap() {
                plan.gestures.push(Gesture::new(t1 - 0.01, t1 + 0.04, 0.015, 0.03, 6.0, Targets::default().with(ControlKey::ABDUCTION, 0.12)).label("murmur"));
                plan.gestures.push(Gesture::new(t1, t1 + 0.035, 0.02, 0.03, 4.0, Targets::default().with(ControlKey::F0, f0 * 0.89)).label("murmur pitch"));
            }
            if s["syllabic"].as_bool().unwrap() && s["accent"].as_bool().unwrap() {
                plan.gestures.push(Gesture::new(t0, t1, 0.05, 0.05, 1.5, Targets::default().with(ControlKey::F0, f0 * (1.0 + accent))).label("pitch"));
            }
        }
        voice.schedule(&plan.gestures);
        end = end.max(plan.end);
    }
    let n = (FS * (end + 0.4)) as usize;
    let mut buf = vec![0f32; n];
    for b in buf.chunks_mut(128) {
        voice.render(b);
    }
    // float WAV, 48 kHz mono
    let mut w = Vec::with_capacity(44 + n * 4);
    let data = (n * 4) as u32;
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + data).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&3u16.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&(FS as u32).to_le_bytes());
    w.extend_from_slice(&(FS as u32 * 4).to_le_bytes());
    w.extend_from_slice(&4u16.to_le_bytes());
    w.extend_from_slice(&32u16.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&data.to_le_bytes());
    let pk = buf.iter().fold(0f32, |m, x| m.max(x.abs()));
    for x in &buf {
        w.extend_from_slice(&(if x.is_finite() { *x } else { 0.0 }).to_le_bytes());
    }
    std::fs::write(out, w).unwrap();
    println!("wrote {out}  {:.2} s  peak {pk:.3}", n as f64 / FS);
}
