//! PIE through CHIMERA: utterances with exact phone durations (from pie_vtl renders) plus the PIE
//! features the IPA reader drops.
//!   cargo run --release -p chimera-cli --example pie -- in.json out.wav male|female [key=value ...]
//!
//! What it adds to the planner's English-tuned score:
//!   ʷ  lips rounded through the stop and into the next sound
//!   ʱ  murmur: the folds part slightly at the release, a short way into the vowel, pitch a little lower
//!   voiceless stops with a short-lag release (the planner aspirates them English-style)
//!   the pitch accent, also on accented syllabic resonants
//! Options (key=value), all with tuned defaults (see DEFAULTS):
//!   map:X=Y            render segment X as Y (e.g. map:e=ɛ)
//!   ov:L:key=v         set articulator/source `key` to v in the planner's gestures labelled L
//!   shift:L=dt         move the planner's gestures labelled L by dt seconds (negative = earlier)
//!   dv:L:key=d         add d to `key` in the planner's gestures labelled L that set it
//!   lab_ap lab_prot lab_str lab_carry lab_lead   the labiovelar rounding
//!   vot asp_ab          voiceless stops: release lag (s) and abduction
//!   mur_ab mur_dur mur_dip   murmur abduction, its length after the release (s), pitch dip (fraction)
//!   src:F=v            voice source field F (breathiness tension brightness roughness drift pressure)
//!   ng_glottis ng_constr ng_teeth   multipliers on the turbulence noise gains (aspiration, constrictions, teeth)
//!   trill_min           lengthen every non-syllabic r (ɾ or r after map:) to at least this (s), taking the time
//!                       from its neighbours (each keeps at least 55 % of its length); 0 = off
//!   trill_n             >0: every non-syllabic r becomes trill_n explicit tip contacts, each a closing then an
//!                       opening gesture (trill_close = closed share of a cycle, trill_open_y = tip height between
//!                       contacts, trill_speed = articulator speed); replaces the planner's own r/ɾ gestures
//!   f0                  base pitch (Hz; 0 = the body's default)
//!   accent              pitch excursion of accented syllables (fraction of the base)
use chimera::anatomy::human::{ArticKey, BODY_FEMALE, BODY_MALE};
use chimera::control::score::{ControlKey, Gesture, Targets};
use chimera::phonetics::Segment;
use chimera::phonetics::planner::{PlanOptions, plan_utterance};
use chimera::voice::human_voice::{HumanVoice, SourcePatch};
use serde_json::Value;
use std::collections::HashMap;

const FS: f64 = 48000.0;
const LEAD: f64 = 0.06; // the planner's breath-onset lead before the first segment

const DEFAULTS: &[(&str, &str)] = &[
    ("accent", "0.2"),
    ("lab_ap", "0.3"),
    ("lab_prot", "1.4"),
    ("lab_str", "3.0"),
    ("lab_carry", "0.03"),
    ("lab_lead", "0.02"),
    ("vot", "0.012"),
    ("asp_ab", "0.22"),
    ("mur_ab", "0.12"),
    ("mur_dur", "0.04"),
    ("mur_dip", "0.11"),
    ("f0", "0"),
    ("trill_min", "0"),
    ("trill_n", "0"),
    ("trill_close", "0.4"),
    ("trill_open_y", "8.9"),
    ("trill_speed", "4"),
    ("ng_glottis", "1"),
    ("ng_constr", "1"),
    ("ng_teeth", "1"),
];

fn key(name: &str) -> ControlKey {
    match name {
        "abduction" => ControlKey::ABDUCTION,
        "closure" => ControlKey::CLOSURE,
        "breath" => ControlKey::BREATH,
        "f0" => ControlKey::F0,
        _ => ControlKey::Artic(ArticKey::from_name(name).unwrap_or_else(|| panic!("unknown key {name}"))),
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let src: Value = serde_json::from_str(&std::fs::read_to_string(&a[1]).unwrap()).unwrap();
    let out = &a[2];
    let body_name = a.get(3).map_or("male", |s| s.as_str());
    let mut opt: HashMap<String, String> = DEFAULTS.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    let mut remap: HashMap<String, String> = HashMap::new();
    let mut overrides: Vec<(String, ControlKey, f64)> = Vec::new();
    let mut deltas: Vec<(String, ControlKey, f64)> = Vec::new();
    let mut src_patch = SourcePatch::default();
    let mut shifts: Vec<(String, f64)> = Vec::new();
    for kv in a.iter().skip(4) {
        let (k, v) = kv.split_once('=').unwrap_or_else(|| panic!("option {kv:?} is not key=value"));
        if let Some(x) = k.strip_prefix("map:") {
            remap.insert(x.to_string(), v.to_string());
        } else if let Some(x) = k.strip_prefix("shift:") {
            shifts.push((x.to_string(), v.parse().unwrap()));
        } else if let Some(x) = k.strip_prefix("src:") {
            let v: f64 = v.parse().unwrap();
            match x {
                "breathiness" => src_patch.breathiness = Some(v),
                "tension" => src_patch.tension = Some(v),
                "brightness" => src_patch.brightness = Some(v),
                "roughness" => src_patch.roughness = Some(v),
                "drift" => src_patch.drift = Some(v),
                "pressure" => src_patch.pressure = Some(v),
                _ => panic!("unknown source field {x}"),
            }
        } else if let Some(x) = k.strip_prefix("dv:") {
            let (label, name) = x.split_once(':').unwrap();
            deltas.push((label.to_string(), key(name), v.parse().unwrap()));
        } else if let Some(x) = k.strip_prefix("ov:") {
            let (label, name) = x.split_once(':').unwrap();
            overrides.push((label.to_string(), key(name), v.parse().unwrap()));
        } else {
            assert!(opt.contains_key(k), "unknown option {k}");
            opt.insert(k.to_string(), v.to_string());
        }
    }
    let f = |k: &str| -> f64 { opt[k].parse().unwrap() };
    let accent = f("accent");
    let (body, mut f0) = if body_name == "female" { (BODY_FEMALE, 200.0) } else { (BODY_MALE, 110.0) };
    if f("f0") > 0.0 {
        f0 = f("f0");
    }
    chimera::init();
    let mut voice = HumanVoice::new(FS, body, 2);
    src_patch.f0 = Some(f0);
    voice.set_source(&src_patch);
    voice.airway.noise_gain.glottis *= f("ng_glottis");
    voice.airway.noise_gain.constriction *= f("ng_constr");
    voice.airway.noise_gain.teeth *= f("ng_teeth");
    let mut end = 0.0f64;
    for u in src["utterances"].as_array().unwrap() {
        let segs_j = u["segs"].as_array().unwrap();
        let start = u["start"].as_f64().unwrap() + 0.3; // 0.3 s of room for the first breath
        let segs: Vec<Segment> = segs_j
            .iter()
            .map(|s| {
                let ipa = s["ipa"].as_str().unwrap();
                Segment {
                    ipa: remap.get(ipa).cloned().unwrap_or_else(|| ipa.to_string()),
                    dur: s["dur"].as_f64(),
                    stress: Some(s["stress"].as_u64().unwrap() as u8),
                    f0: None,
                    word_end: Some(s["word_end"].as_bool().unwrap()),
                }
            })
            .collect();
        let mut segs = segs;
        let tmin = f("trill_min");
        if tmin > 0.0 {
            for i in 0..segs.len() {
                let syl = segs_j[i]["syllabic"].as_bool().unwrap();
                if syl || !(segs[i].ipa == "r" || segs[i].ipa == "ɾ") {
                    continue;
                }
                let d = segs[i].dur.unwrap();
                if d >= tmin {
                    continue;
                }
                let mut need = tmin - d;
                let mut got = 0.0;
                for j in [i.wrapping_sub(1), i + 1] {
                    if j < segs.len() && need > 0.0 {
                        let dj = segs[j].dur.unwrap();
                        let take = (need / 2.0_f64.max(1.0)).min(dj * 0.45).min(need);
                        segs[j].dur = Some(dj - take);
                        got += take;
                    }
                }
                need -= got;
                // the rest from whichever neighbour still has room
                for j in [i.wrapping_sub(1), i + 1] {
                    if j < segs.len() && need > 1e-6 {
                        let dj = segs[j].dur.unwrap();
                        let take = need.min((dj - 0.55 * segs_j[j]["dur"].as_f64().unwrap()).max(0.0));
                        segs[j].dur = Some(dj - take);
                        got += take;
                        need -= take;
                    }
                }
                segs[i].dur = Some(d + got);
            }
        }
        let mut plan = plan_utterance(&segs, &PlanOptions { body: body_name.into(), t0: start - LEAD, f0: Some(f0), accent: Some(accent), ..Default::default() });
        // PIE voiceless stops are plain: the English aspiration gesture becomes a short-lag one
        let vot = f("vot");
        for g in plan.gestures.iter_mut() {
            if g.label_str() == "aspiration" {
                let was_aspirated = g.targets.get(ControlKey::ABDUCTION).unwrap_or(0.0) > 0.3;
                g.targets = Targets::default().with(ControlKey::ABDUCTION, f("asp_ab"));
                if was_aspirated {
                    g.t1 -= (0.05 - vot) * 0.4;
                    g.ramp_out = 0.02;
                } else {
                    g.t1 += (vot - 0.012) * 0.4;
                }
            }
            for (label, dt) in &shifts {
                if g.label_str() == label {
                    g.t0 += dt;
                    g.t1 += dt;
                }
            }
            for (label, k, d) in &deltas {
                if g.label_str() == label {
                    if let Some(x) = g.targets.get(*k) {
                        g.targets.set(*k, x + d);
                    }
                }
            }
            for (label, k, v) in &overrides {
                if g.label_str() == label && (g.targets.has(*k) || g.strength >= 40.0) {
                    g.targets.set(*k, *v);
                }
            }
        }
        let ps = plan.segments.clone();
        let tn = f("trill_n");
        if tn > 0.0 {
            let rs: Vec<usize> = (0..segs.len()).filter(|&i| !segs_j[i]["syllabic"].as_bool().unwrap() && (segs[i].ipa == "r" || segs[i].ipa == "ɾ")).collect();
            if !rs.is_empty() {
                let spans: Vec<(f64, f64)> = rs.iter().map(|&i| (ps[i].t0, ps[i].t1)).collect();
                plan.gestures.retain(|g| !((g.label_str() == "r" || g.label_str() == "ɾ") && g.strength >= 40.0 && spans.iter().any(|(a, b)| g.t1 > a - 0.05 && g.t0 < b + 0.05)));
                for (a, b) in spans {
                    let n = tn.round() as usize;
                    let per = (b - a) / n as f64;
                    let cl = f("trill_close");
                    let sp = f("trill_speed");
                    for k in 0..n {
                        let c0 = a + k as f64 * per;
                        let close = Targets::default().with(ArticKey::TtX, 6.25).with(ArticKey::TtY, 9.7);
                        let open = Targets::default().with(ArticKey::TtX, 6.1).with(ArticKey::TtY, f("trill_open_y"));
                        plan.gestures.push(Gesture::new(c0 - 0.006, c0 + cl * per, 0.004, 0.004, 40.0, close).speed(sp).label("trill"));
                        if k + 1 < n {
                            plan.gestures.push(Gesture::new(c0 + cl * per, c0 + per - 0.006, 0.004, 0.004, 40.0, open).speed(sp).label("trill"));
                        }
                    }
                }
            }
        }
        for (i, s) in segs_j.iter().enumerate() {
            let (t0, t1) = (ps[i].t0, ps[i].t1);
            if s["labial"].as_bool().unwrap() {
                let tg = Targets::default().with(ArticKey::LipAp, f("lab_ap")).with(ArticKey::LipProt, f("lab_prot")).with(ArticKey::LipWidth, 0.065);
                plan.gestures.push(Gesture::new(t0 - f("lab_lead"), t1 + f("lab_carry"), 0.03, 0.03, f("lab_str"), tg).label("labial"));
            }
            if s["breathy"].as_bool().unwrap() {
                let d = f("mur_dur");
                plan.gestures.push(Gesture::new(t1 - 0.01, t1 + d, 0.015, 0.03, 6.0, Targets::default().with(ControlKey::ABDUCTION, f("mur_ab"))).label("murmur"));
                plan.gestures.push(Gesture::new(t1, t1 + d * 0.9, 0.02, 0.03, 4.0, Targets::default().with(ControlKey::F0, f0 * (1.0 - f("mur_dip")))).label("murmur pitch"));
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
    for x in &buf {
        w.extend_from_slice(&(if x.is_finite() { *x } else { 0.0 }).to_le_bytes());
    }
    std::fs::write(out, w).unwrap();
}
