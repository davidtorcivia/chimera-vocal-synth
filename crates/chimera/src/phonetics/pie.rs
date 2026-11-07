//! Proto-Indo-European: reconstructed spellings → segments → a gestural plan.
//!
//! The IPA reader ([`super::ipa`]) is built for living languages and drops what PIE depends on: the
//! labialization of kʷ gʷ gʷʰ, the murmur of bʰ dʰ gʰ, syllabic resonants, the pitch accent and the
//! laryngeals. This module reads both the handbook notation (`*h₂éwsōs`, `*ǵʰéyōm`, `*gʷʰén-ti`) and
//! the narrow IPA used by the wékʷos corpus (`χáw.sos`, `ɟʱéjoːm`, `ɡʷʱénti`), and plans each
//! feature as a gesture of the vocal tract on top of [`plan_utterance`]:
//!
//! - **ʷ**: the lips round through the stop and a little way into the next sound.
//! - **ʰ / ʱ** (murmur): the folds part slightly at the release, a little into the vowel, with the
//!   pitch lower there.
//! - **voiced stops** keep voicing through the closure (the larynx lowers, enlarging the cavity
//!   above the folds), so b d g are not heard as p t k.
//! - **p t k** are plain: a short-lag release, not English aspiration.
//! - **r** is an alveolar trill: explicit tip contacts with releases between them, lengthened to
//!   [`PieOptions::trill_min`] with time taken from its neighbours.
//! - **h₁ h₂ h₃** are [h] [χ] [ʁʷ]; **ḱ ǵ ǵʰ** palatal stops [c ɟ ɟʱ].
//! - the accent is a pitch peak on the accented vowel or syllabic resonant.
//!
//! The defaults were tuned against a phoneme recognizer on PIE sentences (`pie/` in the repository
//! root has the harness and the log).

use super::planner::{PlanOptions, plan_utterance};
use super::{Intonation, Plan, Segment};
use crate::anatomy::human::ArticKey;
use crate::control::score::{ControlKey, Gesture, Targets};
use serde::{Deserialize, Serialize};

/// One PIE segment. `base` is the realization in the engine's inventory (`c` for ḱ, `χ` for h₂,
/// `ʁ` for h₃); the flags carry what the inventory has no symbol for.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct PieSeg {
    pub base: String,
    pub long: bool,
    pub accent: bool,
    pub labial: bool,
    pub breathy: bool,
    pub syllabic: bool,
    pub word_start: bool,
    pub word_end: bool,
    /// duration (s); `None` = the rule-based default ([`default_durations`])
    pub dur: Option<f64>,
}

impl PieSeg {
    fn new(base: &str) -> PieSeg {
        PieSeg { base: base.to_string(), ..Default::default() }
    }
    /// The segment in narrow IPA, e.g. `ɡʷʱ`, `ń̩`, `eː`.
    pub fn label(&self) -> String {
        let mut s = self.base.clone();
        if self.syllabic {
            s.push('\u{329}');
        }
        if self.labial {
            s.push('ʷ');
        }
        if self.breathy {
            s.push('ʱ');
        }
        if self.long {
            s.push('ː');
        }
        if self.accent { format!("ˈ{s}") } else { s }
    }
    pub fn is_vowel(&self) -> bool {
        is_vowel(&self.base)
    }
    pub fn is_nucleus(&self) -> bool {
        self.is_vowel() || self.syllabic
    }
}

/// A phrase: segments said in one breath, with its intonation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct PiePhrase {
    pub segments: Vec<PieSeg>,
    pub question: bool,
}

/// How PIE's features are realized. The defaults are the tuned values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct PieOptions {
    /// pitch excursion of an accented syllable (fraction of the base)
    pub accent: f64,
    /// labiovelar rounding: lip aperture (cm), protrusion, strength, how far it carries into the
    /// next sound and how early it starts (s)
    pub lab_ap: f64,
    pub lab_prot: f64,
    pub lab_str: f64,
    pub lab_carry: f64,
    pub lab_lead: f64,
    /// voiceless stops: release lag (s) and glottal abduction
    pub vot: f64,
    pub asp_ab: f64,
    /// murmur: abduction, its length after the release (s), pitch dip (fraction)
    pub mur_ab: f64,
    pub mur_dur: f64,
    pub mur_dip: f64,
    /// voiced stops: larynx height during the closure (cm, lower = voicing lasts), and strength;
    /// `None` leaves the larynx to the vowels
    pub vstop_larynx: Option<f64>,
    pub vstop_str: f64,
    /// trill: contacts, minimum length (s), closed share of a cycle, tip height between contacts,
    /// articulator speed; `trill_n` 0 keeps the inventory's tap
    pub trill_n: u32,
    pub trill_min: f64,
    pub trill_close: f64,
    pub trill_open_y: f64,
    pub trill_speed: f64,
    /// /i/ is raised (tongue body, cm) and the jaw closed a little, so short /i/ isn't heard as /e/
    pub i_raise: f64,
    pub i_jaw: f64,
    /// /j/ tongue body raise (cm) and /w/ lip aperture (cm)
    pub j_raise: f64,
    pub w_lip_ap: f64,
    /// pause between phrases (s)
    pub pause: f64,
}

impl Default for PieOptions {
    fn default() -> Self {
        PieOptions {
            accent: 0.2,
            lab_ap: 0.5,
            lab_prot: 1.4,
            lab_str: 1.2,
            lab_carry: 0.03,
            lab_lead: 0.02,
            vot: 0.025,
            asp_ab: 0.28,
            mur_ab: 0.07,
            mur_dur: 0.04,
            mur_dip: 0.11,
            vstop_larynx: Some(-0.5),
            vstop_str: 3.0,
            trill_n: 3,
            trill_min: 0.09,
            trill_close: 0.4,
            trill_open_y: 8.6,
            trill_speed: 6.0,
            i_raise: 0.15,
            i_jaw: -0.1,
            j_raise: 0.08,
            w_lip_ap: 0.12,
            pause: 0.3,
        }
    }
}

const VOWELS: &str = "aeiouə";

fn is_vowel(b: &str) -> bool {
    b.chars().count() == 1 && VOWELS.contains(b)
}

fn is_voiced_stop(b: &str) -> bool {
    matches!(b, "b" | "d" | "g" | "ɟ")
}

// ------------------------------------------------------------------------------------- reading

/// Precomposed letters the notations use, as (base, combining mark).
fn decompose(c: char) -> Option<(char, char)> {
    const ACUTE: char = '\u{301}';
    const MACRON: char = '\u{304}';
    const RING: char = '\u{325}';
    Some(match c {
        'á' => ('a', ACUTE),
        'é' => ('e', ACUTE),
        'í' => ('i', ACUTE),
        'ó' => ('o', ACUTE),
        'ú' => ('u', ACUTE),
        'ā' => ('a', MACRON),
        'ē' => ('e', MACRON),
        'ī' => ('i', MACRON),
        'ō' => ('o', MACRON),
        'ū' => ('u', MACRON),
        'ḱ' => ('k', ACUTE),
        'ǵ' => ('g', ACUTE),
        'ŕ' => ('r', ACUTE),
        'ĺ' => ('l', ACUTE),
        'ḿ' => ('m', ACUTE),
        'ń' => ('n', ACUTE),
        'ḗ' => ('e', MACRON), // ē + acute: the acute follows as its own mark only in NFD input
        'ṓ' => ('o', MACRON),
        'ḁ' => ('a', RING),
        _ => return None,
    })
}

/// Read PIE in handbook notation or the corpus's narrow IPA. Phrases break at `‖ | , ; ? !` and
/// newlines (`?` makes a question); words at spaces and `-`; `.` (syllable dots) and `*` are ignored.
pub fn parse_pie(text: &str) -> Vec<PiePhrase> {
    let mut chars: Vec<char> = Vec::with_capacity(text.len());
    for c in text.chars() {
        match decompose(c) {
            Some((b, m)) => {
                chars.push(b);
                chars.push(m);
                if c == 'ḗ' || c == 'ṓ' {
                    chars.push('\u{301}');
                }
            }
            None => chars.push(c),
        }
    }
    let mut phrases: Vec<PiePhrase> = Vec::new();
    let mut cur = PiePhrase::default();
    let mut new_word = true;
    let close = |cur: &mut PiePhrase, phrases: &mut Vec<PiePhrase>, q: bool| {
        if let Some(l) = cur.segments.last_mut() {
            l.word_end = true;
        }
        if !cur.segments.is_empty() {
            cur.question = q;
            phrases.push(std::mem::take(cur));
        }
    };
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        i += 1;
        let push = |cur: &mut PiePhrase, new_word: &mut bool, base: &str| {
            if *new_word {
                if let Some(l) = cur.segments.last_mut() {
                    l.word_end = true;
                }
            }
            let mut s = PieSeg::new(base);
            s.word_start = *new_word;
            *new_word = false;
            cur.segments.push(s);
        };
        match c {
            '‖' | '|' | ',' | ';' | '\n' | '!' | '?' => {
                close(&mut cur, &mut phrases, c == '?');
                new_word = true;
            }
            ' ' | '\t' | '-' | '–' => new_word = true,
            '.' | '*' | '·' | '(' | ')' | 'ˈ' => {}
            '\u{301}' => {
                if let Some(l) = cur.segments.last_mut() {
                    match l.base.as_str() {
                        "k" => l.base = "c".into(),
                        "g" => l.base = "ɟ".into(),
                        _ => l.accent = true,
                    }
                }
            }
            '\u{304}' | 'ː' | ':' => {
                if let Some(l) = cur.segments.last_mut() {
                    l.long = true;
                }
            }
            '\u{325}' | '\u{329}' | '\u{30D}' => {
                if let Some(l) = cur.segments.last_mut() {
                    if matches!(l.base.as_str(), "r" | "l" | "m" | "n") {
                        l.syllabic = true;
                    }
                }
            }
            '\u{32F}' => {
                // non-syllabic i̯ u̯ → j w
                if let Some(l) = cur.segments.last_mut() {
                    match l.base.as_str() {
                        "i" => l.base = "j".into(),
                        "u" => l.base = "w".into(),
                        _ => {}
                    }
                }
            }
            'ʷ' => {
                if let Some(l) = cur.segments.last_mut() {
                    l.labial = true;
                }
            }
            'ʰ' | 'ʱ' => {
                if let Some(l) = cur.segments.last_mut() {
                    l.breathy = true;
                }
            }
            '₁' | '₂' | '₃' | '1' | '2' | '3' => {
                // h₁ h₂ h₃ (also h1 h2 h3)
                if let Some(l) = cur.segments.last_mut() {
                    if l.base == "h" {
                        match c {
                            '₂' | '2' => l.base = "χ".into(),
                            '₃' | '3' => {
                                l.base = "ʁ".into();
                                l.labial = true;
                            }
                            _ => {}
                        }
                    }
                }
            }
            'a' | 'e' | 'i' | 'o' | 'u' | 'ə' | 'p' | 'b' | 't' | 'd' | 'k' | 'c' | 's' | 'z' | 'm' | 'n' | 'l' | 'r' | 'w' | 'j' | 'h' | 'χ' => {
                push(&mut cur, &mut new_word, &c.to_string())
            }
            'g' | 'ɡ' => push(&mut cur, &mut new_word, "g"),
            'ɟ' => push(&mut cur, &mut new_word, "ɟ"),
            'y' => push(&mut cur, &mut new_word, "j"),
            'x' => push(&mut cur, &mut new_word, "χ"),
            'ɣ' | 'ʁ' => push(&mut cur, &mut new_word, "ʁ"),
            'H' => push(&mut cur, &mut new_word, "h"),
            _ => {} // anything else is not PIE
        }
    }
    close(&mut cur, &mut phrases, false);
    // diphthongs written with vowels (*léukā, *déiwos): an unaccented short i or u right after a
    // vowel in the same word is the glide
    for ph in phrases.iter_mut() {
        for k in 1..ph.segments.len() {
            let (prev, s) = (&ph.segments[k - 1], &ph.segments[k]);
            if prev.is_vowel() && !s.word_start && !s.accent && !s.long && (s.base == "i" || s.base == "u") {
                let g = if s.base == "i" { "j" } else { "w" };
                ph.segments[k].base = g.into();
            }
        }
    }
    phrases
}

// ------------------------------------------------------------------------------------- timing

/// Rule-based durations (s) for a phrase without a timeline: a deliberate, recited delivery
/// (pie_vtl's table), accented nuclei ×1.2, the phrase's last nucleus ×1.45 and what follows ×1.2.
pub fn default_durations(segs: &[PieSeg]) -> Vec<f64> {
    let last_nuc = segs.iter().rposition(|s| s.is_nucleus()).unwrap_or(segs.len().saturating_sub(1));
    segs.iter()
        .enumerate()
        .map(|(i, s)| {
            let b = s.base.as_str();
            let mut d = if s.is_vowel() {
                let d = if b == "ə" { 0.055 } else if s.long { 0.19 } else { 0.105 };
                if s.accent { d * 1.2 } else { d }
            } else if s.syllabic {
                0.115 * if s.accent { 1.2 } else { 1.0 }
            } else {
                match b {
                    "p" | "t" | "k" | "c" => 0.085,
                    "b" | "d" | "g" | "ɟ" => 0.07,
                    "χ" | "ʁ" => 0.115,
                    "s" | "z" => 0.095,
                    "h" => 0.07,
                    "m" | "n" => 0.07,
                    "w" | "j" => 0.06,
                    "l" => 0.06,
                    "r" => 0.028,
                    _ => 0.08,
                }
            };
            if i == last_nuc {
                d *= 1.45;
            } else if i > last_nuc {
                d *= 1.2;
            }
            d
        })
        .collect()
}

// ------------------------------------------------------------------------------------- planning

fn is_trill(s: &PieSeg) -> bool {
    s.base == "r" && !s.syllabic
}

/// The engine segment the planner sees for a PIE segment.
fn engine_ipa(s: &PieSeg) -> &str {
    match s.base.as_str() {
        "r" if s.syllabic => "ɹ",
        "r" => "ɾ",
        b => b,
    }
}

/// Plan one phrase. Segments without a duration get [`default_durations`].
pub fn plan_pie_utterance(segs_in: &[PieSeg], question: bool, o: &PlanOptions, p: &PieOptions) -> Plan {
    let n = segs_in.len();
    if n == 0 {
        return Plan { gestures: Vec::new(), segments: Vec::new(), end: o.t0 };
    }
    let defaults = default_durations(segs_in);
    let given: Vec<f64> = segs_in.iter().zip(&defaults).map(|(s, d)| s.dur.unwrap_or(*d)).collect();
    let mut durs = given.clone();
    // a trill needs time for its contacts: take it from the neighbours, each keeping ≥ 55 %
    if p.trill_n > 0 && p.trill_min > 0.0 {
        for i in 0..n {
            if !is_trill(&segs_in[i]) || durs[i] >= p.trill_min {
                continue;
            }
            let mut need = p.trill_min - durs[i];
            let mut got = 0.0;
            let nb = [i.checked_sub(1), if i + 1 < n { Some(i + 1) } else { None }];
            for j in nb.into_iter().flatten() {
                let take = (need / 2.0).min(durs[j] * 0.45).min(need);
                durs[j] -= take;
                got += take;
            }
            need -= got;
            for j in nb.into_iter().flatten() {
                if need > 1e-6 {
                    let take = need.min((durs[j] - 0.55 * given[j]).max(0.0));
                    durs[j] -= take;
                    got += take;
                    need -= take;
                }
            }
            durs[i] += got;
        }
    }
    let segs: Vec<Segment> = segs_in
        .iter()
        .zip(&durs)
        .map(|(s, d)| Segment {
            ipa: engine_ipa(s).to_string(),
            dur: Some(*d),
            stress: Some(if s.is_vowel() && s.accent { 1 } else { 0 }),
            f0: None,
            word_end: Some(s.word_end),
        })
        .collect();
    let po = PlanOptions { accent: Some(p.accent), intonation: Some(if question { Intonation::Question } else { Intonation::Statement }), ..o.clone() };
    let mut plan = plan_utterance(&segs, &po);
    let f0 = o.f0.unwrap_or(110.0);
    // plain voiceless stops; the tuned /i/ /j/ /w/
    for g in plan.gestures.iter_mut() {
        match g.label_str() {
            "aspiration" => {
                let was_aspirated = g.targets.get(ControlKey::ABDUCTION).unwrap_or(0.0) > 0.3;
                g.targets = Targets::default().with(ControlKey::ABDUCTION, p.asp_ab);
                if was_aspirated {
                    g.t1 -= (0.05 - p.vot) * 0.4;
                    g.ramp_out = 0.02;
                } else {
                    g.t1 += (p.vot - 0.012) * 0.4;
                }
            }
            "i" => {
                if let Some(y) = g.targets.get(ArticKey::TbY) {
                    g.targets.set(ArticKey::TbY, y + p.i_raise);
                }
                if let Some(j) = g.targets.get(ArticKey::Jaw) {
                    g.targets.set(ArticKey::Jaw, j + p.i_jaw);
                }
            }
            "j" => {
                if let Some(y) = g.targets.get(ArticKey::TbY) {
                    g.targets.set(ArticKey::TbY, y + p.j_raise);
                }
            }
            "w" => {
                if g.targets.has(ArticKey::LipAp) || g.strength >= 40.0 {
                    g.targets.set(ArticKey::LipAp, p.w_lip_ap);
                }
            }
            _ => {}
        }
    }
    let ps = plan.segments.clone();
    // the trill: explicit contacts replace the tap's gestures
    if p.trill_n > 0 {
        let spans: Vec<(f64, f64)> = (0..n).filter(|&i| is_trill(&segs_in[i])).map(|i| (ps[i].t0, ps[i].t1)).collect();
        if !spans.is_empty() {
            plan.gestures.retain(|g| !(g.label_str() == "ɾ" && g.strength >= 40.0 && spans.iter().any(|(a, b)| g.t1 > a - 0.05 && g.t0 < b + 0.05)));
            for (a, b) in spans {
                let k_n = p.trill_n as usize;
                let per = (b - a) / k_n as f64;
                for k in 0..k_n {
                    let c0 = a + k as f64 * per;
                    let close = Targets::default().with(ArticKey::TtX, 6.25).with(ArticKey::TtY, 9.7);
                    plan.gestures.push(Gesture::new(c0 - 0.006, c0 + p.trill_close * per, 0.004, 0.004, 40.0, close).speed(p.trill_speed).label("r"));
                    if k + 1 < k_n {
                        let open = Targets::default().with(ArticKey::TtX, 6.1).with(ArticKey::TtY, p.trill_open_y);
                        plan.gestures.push(Gesture::new(c0 + p.trill_close * per, c0 + per - 0.006, 0.004, 0.004, 40.0, open).speed(p.trill_speed).label("r"));
                    }
                }
            }
        }
    }
    for (i, s) in segs_in.iter().enumerate() {
        let (t0, t1) = (ps[i].t0, ps[i].t1);
        if s.labial {
            let tg = Targets::default().with(ArticKey::LipAp, p.lab_ap).with(ArticKey::LipProt, p.lab_prot).with(ArticKey::LipWidth, 0.065);
            plan.gestures.push(Gesture::new(t0 - p.lab_lead, t1 + p.lab_carry, 0.03, 0.03, p.lab_str, tg).label("ʷ"));
        }
        if is_voiced_stop(&s.base) {
            if let Some(y) = p.vstop_larynx {
                plan.gestures.push(Gesture::new(t0 - 0.01, t1 - 0.005, 0.02, 0.02, p.vstop_str, Targets::default().with(ArticKey::Larynx, y)).label("voice bar"));
            }
        }
        if s.breathy {
            plan.gestures.push(Gesture::new(t1 - 0.01, t1 + p.mur_dur, 0.015, 0.03, 6.0, Targets::default().with(ControlKey::ABDUCTION, p.mur_ab)).label("ʱ"));
            plan.gestures.push(Gesture::new(t1, t1 + p.mur_dur * 0.9, 0.02, 0.03, 4.0, Targets::default().with(ControlKey::F0, f0 * (1.0 - p.mur_dip))).label("pitch"));
        }
        if s.syllabic && s.accent {
            plan.gestures.push(Gesture::new(t0, t1, 0.05, 0.05, 1.5, Targets::default().with(ControlKey::F0, f0 * (1.0 + p.accent))).label("pitch"));
        }
    }
    // the planned segments carry the PIE labels (ḱ shows as c, h₂ as χ; flags as diacritics)
    for (ps, s) in plan.segments.iter_mut().zip(segs_in) {
        ps.ipa = s.label();
    }
    plan
}

/// Plan PIE text (see [`parse_pie`]), phrase after phrase from `o.t0`.
pub fn plan_pie(text: &str, o: &PlanOptions, p: &PieOptions) -> Plan {
    plan_pie_phrases(&parse_pie(text), o, p)
}

/// Plan parsed phrases, [`PieOptions::pause`] apart.
pub fn plan_pie_phrases(phrases: &[PiePhrase], o: &PlanOptions, p: &PieOptions) -> Plan {
    let mut all = Plan { gestures: Vec::new(), segments: Vec::new(), end: o.t0 };
    let mut t = o.t0;
    for ph in phrases {
        let plan = plan_pie_utterance(&ph.segments, ph.question, &PlanOptions { t0: t, ..o.clone() }, p);
        all.gestures.extend_from_slice(&plan.gestures);
        all.segments.extend(plan.segments);
        t = plan.end + p.pause;
        all.end = plan.end;
    }
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(t: &str) -> Vec<String> {
        parse_pie(t).iter().flat_map(|p| p.segments.iter().map(|s| s.label())).collect()
    }

    #[test]
    fn reads_handbook_notation() {
        assert_eq!(labels("*h₂éwsōs"), ["χ", "ˈe", "w", "s", "oː", "s"]);
        assert_eq!(labels("*ǵʰéyōm"), ["ɟʱ", "ˈe", "j", "oː", "m"]);
        assert_eq!(labels("gʷʰén-ti"), ["gʷʱ", "ˈe", "n", "t", "i"]);
        assert_eq!(labels("h₃ékʷ"), ["ʁʷ", "ˈe", "kʷ"]);
        assert_eq!(labels("ḱm̥tóm"), ["c", "m̩", "t", "ˈo", "m"]);
        assert_eq!(labels("dʰugh₂tḗr"), ["dʱ", "u", "g", "χ", "t", "ˈeː", "r"]);
        assert_eq!(labels("*léukā déiwos"), ["l", "ˈe", "w", "k", "aː", "d", "ˈe", "j", "w", "o", "s"]);
    }

    #[test]
    fn reads_corpus_ipa() {
        assert_eq!(labels("χáw.sos"), ["χ", "ˈa", "w", "s", "o", "s"]);
        assert_eq!(labels("ń̩dʱ.ɡʷʱi.tom"), ["ˈn̩", "dʱ", "gʷʱ", "i", "t", "o", "m"]);
        assert_eq!(labels("ɣʷkʷbʱós"), ["ʁʷ", "kʷ", "bʱ", "ˈo", "s"]);
        assert_eq!(labels("wih.róːs"), ["w", "i", "h", "r", "ˈoː", "s"]);
    }

    #[test]
    fn phrases_and_words() {
        let p = parse_pie("χáwsos léwka ‖ nu céjwa?");
        assert_eq!(p.len(), 2);
        assert!(p[1].question && !p[0].question);
        let ends: Vec<bool> = p[0].segments.iter().map(|s| s.word_end).collect();
        assert_eq!(ends, [false, false, false, false, false, true, false, false, false, false, true]);
    }

    #[test]
    fn plans_every_feature() {
        let o = PlanOptions { body: crate::anatomy::human::BodyMix::MALE, t0: 0.0, f0: Some(110.0), ..Default::default() };
        let plan = plan_pie("gʷʰénti bʰréh₂tēr", &o, &PieOptions::default());
        let has = |l: &str| plan.gestures.iter().any(|g| g.label_str() == l);
        assert!(has("ʷ") && has("ʱ") && has("voice bar") && has("r"));
        assert!(!plan.gestures.iter().any(|g| g.label_str() == "ɾ" && g.strength >= 40.0), "the tap is replaced by the trill");
        // the trill got its minimum length
        let r = plan.segments.iter().find(|s| s.ipa == "r").unwrap();
        assert!(r.t1 - r.t0 >= 0.089, "trill {}", r.t1 - r.t0);
        assert_eq!(plan.segments.len(), 12);
    }
}
