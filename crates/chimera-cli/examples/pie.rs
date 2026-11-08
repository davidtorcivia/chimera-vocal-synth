//! PIE through CHIMERA on given phone timings (pie_vtl renders), for the recognizer harness in `pie/`.
//!   cargo run --release -p chimera-cli --example pie -- in.json out.wav male|female [key=value ...]
//!
//! in.json: {"utterances": [{"start": s, "segs": [{"pie": "ɡʷʱ", "dur": s, "accent", "labial", "breathy",
//! "syllabic", "long", "word_start", "word_end"}, ...]}, ...]} (pie/eval.py writes it). Each utterance is
//! planned by `chimera::phonetics::pie::plan_pie_utterance` at start + 0.3 s.
//!
//! Options (key=value):
//!   any PieOptions field, snake_case (lab_ap=0.5, trill_n=3, vstop_larynx=none, ...)
//!   src:F=v       voice source field F (breathiness tension brightness roughness drift pressure)
//!   ng_glottis ng_constr ng_teeth   multipliers on the turbulence noise gains
//!   f0=Hz         base pitch (default 110 male, 200 female)
//!   map:X=Y       PIE base X planned as Y (e.g. map:e=ɛ)
//!   ov:L:key=v    set articulator/source `key` to v in the gestures labelled L
//!   dv:L:key=d    add d to `key` in the gestures labelled L that set it
//!   shift:L=dt    move the gestures labelled L by dt seconds
use chimera::anatomy::human::{ArticKey, BODY_FEMALE, BODY_MALE};
use chimera::control::score::ControlKey;
use chimera::phonetics::pie::{LineSpan, PieOptions, PieSeg, plan_pie_utterance_in};
use chimera::phonetics::planner::PlanOptions;
use chimera::voice::human_voice::{HumanVoice, SourcePatch};
use serde_json::{Map, Value};
use std::collections::HashMap;

const FS: f64 = 48000.0;
const LEAD: f64 = 0.06; // the planner's breath-onset lead before the first segment

fn key(name: &str) -> ControlKey {
    match name {
        "abduction" => ControlKey::ABDUCTION,
        "closure" => ControlKey::CLOSURE,
        "breath" => ControlKey::BREATH,
        "f0" => ControlKey::F0,
        _ => ControlKey::Artic(ArticKey::from_name(name).unwrap_or_else(|| panic!("unknown key {name}"))),
    }
}

fn camel(s: &str) -> String {
    let mut out = String::new();
    let mut up = false;
    for c in s.chars() {
        if c == '_' {
            up = true;
        } else if up {
            out.extend(c.to_uppercase());
            up = false;
        } else {
            out.push(c);
        }
    }
    out
}

fn base(ph: &str) -> String {
    let b: String = ph.chars().filter(|c| !"ʷʱʰ\u{329}\u{325}ːˈ\u{301}".contains(*c)).collect();
    match b.as_str() {
        "ɡ" => "g".into(),
        "ɣ" => "ʁ".into(),
        _ => b,
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let src: Value = serde_json::from_str(&std::fs::read_to_string(&a[1]).unwrap()).unwrap();
    let out = &a[2];
    let body_name = a.get(3).map_or("male", |s| s.as_str());
    let mut pie_opts = Map::new();
    let mut remap: HashMap<String, String> = HashMap::new();
    let mut overrides: Vec<(String, ControlKey, f64)> = Vec::new();
    let mut deltas: Vec<(String, ControlKey, f64)> = Vec::new();
    let mut shifts: Vec<(String, f64)> = Vec::new();
    let mut src_patch = SourcePatch::default();
    let mut ng = [1.0f64; 3];
    let mut f0_opt = 0.0;
    for kv in a.iter().skip(4) {
        let (k, v) = kv.split_once('=').unwrap_or_else(|| panic!("option {kv:?} is not key=value"));
        if let Some(x) = k.strip_prefix("map:") {
            remap.insert(x.to_string(), v.to_string());
        } else if let Some(x) = k.strip_prefix("shift:") {
            shifts.push((x.to_string(), v.parse().unwrap()));
        } else if let Some(x) = k.strip_prefix("dv:") {
            let (label, name) = x.split_once(':').unwrap();
            deltas.push((label.to_string(), key(name), v.parse().unwrap()));
        } else if let Some(x) = k.strip_prefix("ov:") {
            let (label, name) = x.split_once(':').unwrap();
            overrides.push((label.to_string(), key(name), v.parse().unwrap()));
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
        } else if k == "ng_glottis" {
            ng[0] = v.parse().unwrap();
        } else if k == "ng_constr" {
            ng[1] = v.parse().unwrap();
        } else if k == "ng_teeth" {
            ng[2] = v.parse().unwrap();
        } else if k == "f0" {
            f0_opt = v.parse().unwrap();
        } else {
            let val = if v == "none" { Value::Null } else { serde_json::from_str::<Value>(v).unwrap_or_else(|_| panic!("bad value {kv}")) };
            pie_opts.insert(camel(k), val);
        }
    }
    let p: PieOptions = serde_json::from_value(Value::Object(pie_opts.clone())).unwrap_or_else(|e| panic!("options {pie_opts:?}: {e}"));
    // serde ignores unknown keys: catch typos
    let known = serde_json::to_value(PieOptions::default()).unwrap();
    for k in pie_opts.keys() {
        assert!(known.get(k).is_some(), "unknown option {k}");
    }
    let (body, mut f0) = if body_name == "female" { (BODY_FEMALE, 200.0) } else { (BODY_MALE, 110.0) };
    if f0_opt > 0.0 {
        f0 = f0_opt;
    }
    chimera::init();
    let mut voice = HumanVoice::new(FS, body, 2);
    // the PIE voice (source and noise settings), then any src: overrides on top
    chimera::phonetics::pie::setup_voice(&mut voice, f0, &p);
    src_patch.f0 = Some(f0);
    voice.set_source(&src_patch);
    voice.airway.noise_gain.glottis *= ng[0];
    voice.airway.noise_gain.constriction *= ng[1];
    voice.airway.noise_gain.teeth *= ng[2];
    let mut end = 0.0f64;
    for u in src["utterances"].as_array().unwrap() {
        let start = u["start"].as_f64().unwrap() + 0.3; // 0.3 s of room for the first breath
        let segs: Vec<PieSeg> = u["segs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| {
                let b = base(s["pie"].as_str().unwrap());
                let fl = |k: &str| s[k].as_bool().unwrap_or(false);
                PieSeg {
                    base: remap.get(&b).cloned().unwrap_or(b),
                    long: fl("long"),
                    accent: fl("accent"),
                    labial: fl("labial"),
                    breathy: fl("breathy"),
                    syllabic: fl("syllabic"),
                    word_start: fl("word_start"),
                    word_end: fl("word_end"),
                    dur: s["dur"].as_f64(),
                }
            })
            .collect();
        let o = PlanOptions { body: body_name.into(), t0: start - LEAD, f0: Some(f0), ..Default::default() };
        // the line this breath group belongs to (eval.py adds it), shifted like the group
        let line = u.get("line_start").and_then(|x| x.as_f64()).map(|ls| LineSpan {
            start: ls + 0.3,
            end: u["line_end"].as_f64().unwrap() + 0.3,
            ends_line: u["ends_line"].as_bool().unwrap_or(true),
        });
        let mut plan = plan_pie_utterance_in(&segs, false, line, &o, &p);
        for g in plan.gestures.iter_mut() {
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
