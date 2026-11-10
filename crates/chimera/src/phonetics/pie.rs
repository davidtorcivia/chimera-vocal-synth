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
    /// The segment in narrow IPA with PIE's acute for the accent, e.g. `gʷʱ`, `ń̩`, `éː`.
    pub fn label(&self) -> String {
        let mut s = self.base.clone();
        if self.accent {
            s.push('\u{301}'); // PIE's own accent mark: é, ń̩
        }
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
        s
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
    /// a labialized consonant before a vowel releases through a [w] this long (s), taken from the
    /// vowel; 0 = off
    pub lab_glide: f64,
    /// a palatal stop (ḱ ǵ ǵʰ) before a vowel releases through a [j] this long (s), taken from the vowel:
    /// the palatalized release [kʲ] of the palatovelars; 0 = off
    pub pal_glide: f64,
    /// a palatal stop before a vowel releases through palatal frication this long (s; [cç], voiced [ɟʝ]):
    /// the slow, fricated release of a long dorsal–palatal contact; 0 = off (pal_glide wins if both are set)
    pub pal_fric: f64,
    /// voiceless stops: release lag (s) and glottal abduction
    pub vot: f64,
    pub asp_ab: f64,
    /// share of every vowel's span (centred) kept free of the neighbouring consonants' constriction
    /// gestures: their lead and ramps are trimmed so the tract opens for even a short vowel (the
    /// inventory's ~60 ms anticipation swallowed a 55 ms schwa between s and t whole); 0 = off
    /// how long before a nasal the velum starts to lower (s, negative: the planner's 70 ms, English-like) and its
    /// ramp; 20 ms over 20 ms keeps the vowel before a nasal oral and helps syllabic nasals
    pub velum_lead: f64,
    pub velum_ramp: f64,
    /// an obstruent before h is fully released by this share of the h's span (negative: off)
    pub h_release: f64,
    pub vowel_open: f64,
    /// only vowels shorter than this (s) are guarded (long vowels leave room for normal anticipation)
    pub vowel_open_max: f64,
    /// the glottal opening of voiceless sounds (the planner's "aspiration", "voiceless" and "h" gestures):
    /// strength (the planner's 5–6 saturate the opening, so it holds through most of the fade and voicing
    /// came back ~30 ms late) and how much earlier it ends (s)
    pub devoice_str: f64,
    /// strength of a voiceless stop's opening (its release; `devoice_str` is the fricatives' and h's)
    pub devoice_str_stop: f64,
    pub devoice_early: f64,
    /// how fast the opening comes in (s; the planner's 0.03–0.035 starts it well before the closure, so a
    /// short vowel between voiceless sounds lost its voicing from both sides); 0 = the planner's
    pub devoice_ramp_in: f64,
    /// how much earlier a voiceless fricative's opening ends (s): less than a stop's, or its tail voices
    /// (s heard as z)
    pub devoice_early_fric: f64,
    /// murmur: abduction, its length after the release (s), pitch dip (fraction)
    pub mur_ab: f64,
    pub mur_dur: f64,
    pub mur_dip: f64,
    /// murmur source channel level (0..1): breathy voicing that keeps the folds vibrating
    pub mur_level: f64,
    /// what full murmur does to the glottis (see HumanVoice::murmur_chink and friends)
    pub mur_chink: f64,
    pub mur_oq: f64,
    /// voiced stops: larynx height during the closure (cm, lower = voicing lasts), and strength;
    /// `None` leaves the larynx to the vowels
    pub vstop_larynx: Option<f64>,
    pub vstop_str: f64,
    /// voiced stops: tongue root advance during the closure (cm; `None` leaves it): with the lowered larynx,
    /// the other way speakers enlarge the cavity behind a closure so voicing lasts
    pub vstop_root: Option<f64>,
    /// trill: contacts, minimum length (s), closed share of a cycle, tip height between contacts,
    /// articulator speed; `trill_n` 0 keeps the inventory's tap
    pub trill_n: u32,
    pub trill_min: f64,
    pub trill_close: f64,
    pub trill_open_y: f64,
    /// tongue-tip height of a trill contact (higher = a firmer, more complete closure)
    pub trill_close_y: f64,
    pub trill_speed: f64,
    /// /i/ is raised (tongue body, cm) and the jaw closed a little, so short /i/ isn't heard as /e/
    /// t d n (and dʰ): tongue tip forward (cm) of the inventory's alveolar closure: PIE's dental-alveolar
    /// stops (the inventory's tip sits under the crest of the ridge, heard as retroflex [ʈ] at times)
    pub coronal_front: f64,
    /// tongue body for l (None: whatever the neighbours leave); a back, lowered body is a dark l
    pub l_tb_x: Option<f64>,
    pub l_tb_y: Option<f64>,
    /// tongue body across r and r̥ (None: the planner's weak coronal shape), held at strength r_tb_str
    pub r_tb_x: Option<f64>,
    pub r_tb_y: Option<f64>,
    pub r_tb_str: f64,
    /// ḱ ǵ ǵʰ: tongue body forward (cm) of the inventory's palatal stop, whose contact sits near the
    /// middle of the hard palate (a fronted [k]); a true [c] closes further forward
    pub c_front: f64,
    pub i_raise: f64,
    pub i_jaw: f64,
    /// /e/ tongue body height change (cm)
    pub e_raise: f64,
    /// PIE *a planned as this engine vowel ("a" or "ɑ": the engine's calibrated [ɑ] meets the reference
    /// formants of an open central-back [a] better than its [a], whose F1 falls short)
    pub a_vowel: String,
    /// /a/ jaw opening change (cm), /u/ and /o/ tongue body front–back change (cm; − = back)
    pub a_jaw: f64,
    pub u_back: f64,
    /// lip aperture for u (None: the inventory's); tighter lips give u its low F1 without the tongue
    /// nearly closing on the velum (which silenced and devoiced it)
    pub u_lip_ap: Option<f64>,
    pub o_back: f64,
    /// further per-vowel articulator changes, "vowel:key:delta" (e.g. "ɑ:tbY:-0.3", "ɑ:root:-0.2")
    pub vowel_deltas: Vec<String>,
    /// /j/ tongue body raise (cm)
    pub j_raise: f64,
    /// a real /w/: lip aperture (cm), protrusion, tongue body front–back (cm)
    pub w_lip_ap: f64,
    pub w_lip_prot: f64,
    pub w_tb_x: f64,
    /// a real /w/: tongue body height change (cm); the back of the tongue must stay clear of the
    /// soft palate (an approximant, ~0.3 cm² or more), or it hisses
    pub w_tb_y: f64,
    /// the [w] release of a labiovelar: lip aperture (cm)
    pub glide_lip_ap: f64,
    /// h₂ [χ]: tongue body height change at the uvula (cm, + narrower) and the glottal opening
    /// during it (the planner's 0.3 for any voiceless fricative makes it mostly breath, an [h])
    pub chi_raise: f64,
    pub chi_abd: f64,
    /// h₃ [ʁʷ]: tongue body height change (cm, + narrower: more frication)
    pub gh_raise: f64,
    /// gain of the dorsal obstacle noise source (the voice's `airway.noise_gain.dorsal`; see
    /// [`setup_voice`]): what makes [χ] a loud uvular fricative rather than an [h]
    pub dorsal_noise: f64,
    /// upper band edge (Hz) of the sibilant (teeth) noise source; the engine's default is 5000
    pub sibilant_lowpass: f64,
    /// /s/: tongue tip forward (cm) and lips spread (their protrusion target; `None` leaves the lips
    /// to the vowels), which shorten the cavity in front of the constriction and raise its resonance
    pub s_tip_fwd: f64,
    pub s_lip_prot: Option<f64>,
    /// /s/: jaw opening target (`None`: the inventory's 0.3) and tongue tip height change (cm)
    pub s_jaw: Option<f64>,
    pub s_tip_up: f64,
    /// Reynolds threshold of the dorsal source (the engine's general one is 1800)
    pub dorsal_re: f64,
    /// how far downstream of the constriction the dorsal source enters (cm)
    pub dorsal_offset: f64,
    /// a second injection point (cm; 0 = none) with `dorsal_split` of the gain (see NoiseGains)
    pub dorsal_offset2: f64,
    pub dorsal_split: f64,
    /// band edge (Hz) of the dorsal source, and a four-pole (steep) low-pass instead of two-pole. χ has a
    /// sweet spot: darker is heard as [k] or nothing, brighter (a 4.5 kHz edge) as [s]
    pub dorsal_lowpass: f64,
    pub dorsal_steep: bool,
    /// the narrowing (cm²) above which the dorsal source has faded out (from 0.4× this)
    pub dorsal_area: f64,
    /// upper band edge (Hz) of the glottal aspiration noise (the engine's default 3500)
    pub glottis_lowpass: f64,
    /// the narrowing (cm²) above which the general constriction source has faded out (the engine's 0.9 lets
    /// laterals and close vowels hiss)
    pub constr_area: f64,
    /// stop-release burst: upper band edge (Hz; 0 = off), gain multiplier and duration (s) of the constriction
    /// source just after a full closure opens (see NoiseGains::burst_lowpass)
    pub burst_lowpass: f64,
    pub burst_gain: f64,
    pub burst_time: f64,
    /// the burst gain at the lips and at the tongue tip (`burst_gain` is the tongue body's)
    pub burst_gain_lab: f64,
    pub burst_gain_front: f64,
    /// multiplier on the glottal aspiration noise gain
    pub glottis_noise: f64,
    /// aspiration colour (band edge, Hz) during h₁ [h] and h₂ [χ], set by gestures (the aspLowpass
    /// channel); 0 = leave the voice's own. A darker breath keeps [h] in the vowel's formants and gives
    /// χ the low-centred noise of a uvular, while stop releases and sibilants keep the voice's own colour
    pub h_asp_lowpass: f64,
    pub chi_asp_lowpass: f64,
    /// syllabic r̥: 0 = an [ɹ̩]-like approximant; n > 0 = a vocoid with n trill contacts across its middle
    pub syl_r_trill: u32,
    /// the PIE pitch model (pie_vtl's): replaces the planner's English intonation with a line baseline
    /// falling from `decl_start` to `decl_end` (st re the base pitch), a fall of `final_fall` st through
    /// the line's last word (a rise of `question_rise` for questions), and on every accented nucleus a
    /// raised-cosine bump of `accent_st` st (rising over `accent_rise` s, falling over `accent_fall` s;
    /// overlapping bumps do not stack), with a dip of `breathy_dip_st` after a murmured stop
    pub pie_pitch: bool,
    pub accent_st: f64,
    pub decl_start: f64,
    pub decl_end: f64,
    pub final_fall: f64,
    pub question_rise: f64,
    pub accent_rise: f64,
    pub accent_fall: f64,
    pub breathy_dip_st: f64,
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
            lab_glide: 0.01,
            pal_glide: 0.0,
            pal_fric: 0.0,
            vot: 0.015,
            asp_ab: 0.22,
            velum_lead: 0.02,
            velum_ramp: 0.02,
            h_release: -1.0,
            vowel_open: 0.3,
            vowel_open_max: 1.0,
            devoice_str: 1.5,
            devoice_str_stop: 2.5,
            devoice_early: 0.015,
            devoice_ramp_in: 0.0,
            devoice_early_fric: 0.015,
            mur_ab: 0.0,
            mur_dur: 0.06,
            mur_dip: 0.11,
            mur_level: 0.7,
            mur_chink: 0.1,
            mur_oq: 0.2,
            vstop_larynx: Some(-1.0),
            vstop_str: 6.0,
            vstop_root: Some(0.8),
            trill_n: 3,
            trill_min: 0.11,
            trill_close: 0.4,
            trill_open_y: 8.6,
            trill_close_y: 9.7,
            trill_speed: 6.0,
            c_front: 0.0,
            coronal_front: 0.2,
            l_tb_x: Some(2.0),
            l_tb_y: Some(5.4),
            r_tb_x: None,
            r_tb_y: None,
            r_tb_str: 6.0,
            i_raise: 0.22,
            i_jaw: -0.1,
            e_raise: 0.1,
            a_vowel: "ɑ".into(),
            a_jaw: 0.0,
            u_back: -0.2,
            u_lip_ap: Some(0.16),
            o_back: 0.0,
            vowel_deltas: vec!["ɑ:tbY:-0.3".into()],
            j_raise: 0.08,
            w_lip_ap: 0.1,
            w_lip_prot: 1.5,
            w_tb_x: 2.1,
            w_tb_y: -0.08,
            glide_lip_ap: 0.12,
            chi_raise: 0.0,
            chi_abd: 0.3,
            gh_raise: 0.0,
            syl_r_trill: 2,
            dorsal_noise: 4.8e-5,
            sibilant_lowpass: 5000.0,
            dorsal_re: 5500.0,
            dorsal_offset: 1.6,
            dorsal_offset2: 0.0,
            dorsal_split: 0.35,
            dorsal_area: 0.6,
            dorsal_lowpass: 900.0,
            dorsal_steep: false,
            glottis_lowpass: 3500.0,
            glottis_noise: 1.0,
            constr_area: 0.45,
            // stop-release bursts (2026-11-10): Chimera's releases were low pops (centroid p 283 Hz, t 1.4 kHz,
            // k 0.7 kHz; the owner heard a low click through the converter). These put them at the paid
            // conversion's brightness on the held-out lines (p 1160 / 1155 Hz, t 3142 / 2968, k 3653 / 3632,
            // kʷ 2120 / 2219, c 2432 / 3137) with the recognizer's misses unchanged (±0.2 points)
            burst_lowpass: 8000.0,
            burst_gain: 6.0,
            burst_time: 0.02,
            burst_gain_lab: 10.0,
            burst_gain_front: 2.0,
            h_asp_lowpass: 1200.0,
            chi_asp_lowpass: 0.0,
            s_tip_fwd: 0.0,
            s_lip_prot: Some(-0.3),
            s_jaw: None,
            s_tip_up: 0.0,
            pie_pitch: true,
            accent_st: 4.0,
            decl_start: 1.5,
            decl_end: -2.5,
            final_fall: 2.0,
            question_rise: 3.0,
            accent_rise: 0.12,
            accent_fall: 0.15,
            breathy_dip_st: 2.0,
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
fn engine_ipa<'a>(s: &'a PieSeg, p: &'a PieOptions) -> &'a str {
    match s.base.as_str() {
        "r" if s.syllabic && p.syl_r_trill > 0 => "ə",
        "r" if s.syllabic => "ɹ",
        "r" => "ɾ",
        "a" => p.a_vowel.as_str(),
        b => b,
    }
}

/// The line a breath group belongs to, for the pitch model: declination runs across the whole line, and
/// the final fall (or question rise) belongs to the group that ends it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LineSpan {
    pub start: f64,
    pub end: f64,
    pub ends_line: bool,
}

/// Plan one phrase. Segments without a duration get [`default_durations`]. The phrase is its own line.
pub fn plan_pie_utterance(segs_in: &[PieSeg], question: bool, o: &PlanOptions, p: &PieOptions) -> Plan {
    plan_pie_utterance_in(segs_in, question, None, o, p)
}

/// Plan one breath group of a line (`line` in plan time; `None`: the group is the whole line).
pub fn plan_pie_utterance_in(segs_in: &[PieSeg], question: bool, line: Option<LineSpan>, o: &PlanOptions, p: &PieOptions) -> Plan {
    let mut plan = plan_glides(segs_in, question, o, p);
    if p.pie_pitch && !plan.segments.is_empty() {
        pie_pitch(&mut plan, segs_in, question, line, o, p);
    }
    plan
}

/// Replace the planner's intonation with the PIE pitch model (see [`PieOptions::pie_pitch`]).
fn pie_pitch(plan: &mut Plan, segs: &[PieSeg], question: bool, line: Option<LineSpan>, o: &PlanOptions, p: &PieOptions) {
    let ps = &plan.segments;
    let n = ps.len().min(segs.len());
    let line = line.unwrap_or(LineSpan { start: ps[0].t0, end: ps[n - 1].t1, ends_line: true });
    let mut accents: Vec<(f64, f64)> = Vec::new();
    let mut dips: Vec<f64> = Vec::new();
    for i in 0..n {
        if segs[i].is_nucleus() && segs[i].accent {
            accents.push((ps[i].t0, ps[i].t1));
        }
        if segs[i].is_nucleus() && i > 0 && segs[i - 1].breathy {
            dips.push(ps[i].t0);
        }
    }
    // the line's last word, when this group ends the line
    let final_word = if line.ends_line {
        let k = (0..n).rev().find(|&i| segs[i].word_start).unwrap_or(0);
        Some((ps[k].t0, ps[n - 1].t1))
    } else {
        None
    };
    let rc = |x: f64| {
        let x = x.clamp(0.0, 1.0);
        0.5 - 0.5 * (std::f64::consts::PI * x).cos()
    };
    let f = |t: f64| -> f64 {
        let frac = ((t - line.start) / (line.end - line.start).max(1e-3)).clamp(0.0, 1.0);
        let mut v = p.decl_start + (p.decl_end - p.decl_start) * frac;
        if let Some((w0, w1)) = final_word {
            if t > w0 {
                let x = rc((t - w0) / (w1 - w0).max(1e-3));
                v += if question { p.question_rise * x } else { -p.final_fall * x };
            }
        }
        let mut bump: f64 = 0.0;
        for &(a0, a1) in &accents {
            if t < a0 - p.accent_rise || t > a1 + p.accent_fall {
                continue;
            }
            let b = if t < a0 {
                rc((t - (a0 - p.accent_rise)) / p.accent_rise)
            } else if t <= a1 {
                1.0
            } else {
                1.0 - rc((t - a1) / p.accent_fall)
            };
            bump = bump.max(b);
        }
        v += p.accent_st * bump;
        for &d0 in &dips {
            if t >= d0 - 0.01 && t <= d0 + 0.045 {
                v -= p.breathy_dip_st * (std::f64::consts::PI * (t - d0 + 0.01) / 0.055).sin();
            }
        }
        v
    };
    let f0 = o.f0.unwrap_or(110.0);
    plan.gestures.retain(|g| g.label_str() != "pitch");
    let (t_a, t_b) = (ps[0].t0 - 0.06, ps[n - 1].t1 + 0.06);
    let step = 0.015;
    let mut t = t_a;
    while t < t_b {
        let hz = f0 * 2f64.powf(f(t + 0.5 * step) / 12.0);
        plan.gestures.push(Gesture::new(t, t + step, 0.008, 0.008, 3.0, Targets::default().with(ControlKey::F0, hz)).label("pitch"));
        t += step;
    }
}

fn plan_glides(segs_in: &[PieSeg], question: bool, o: &PlanOptions, p: &PieOptions) -> Plan {
    let palatal = |s: &PieSeg| s.base == "c" || s.base == "ɟ";
    let wants = |s: &PieSeg| (p.lab_glide > 0.0 && s.labial) || ((p.pal_glide > 0.0 || p.pal_fric > 0.0) && palatal(s));
    if !segs_in.iter().any(wants) {
        return plan_core(segs_in, &[], question, o, p);
    }
    // a labialized consonant releases into its vowel through a rounded transition: plan it as a
    // short [w] at the vowel's start (time taken from the vowel), then fold that span back into the
    // consonant, so the plan has one segment per PIE segment
    let defaults = default_durations(segs_in);
    let mut segs: Vec<PieSeg> = Vec::with_capacity(segs_in.len() + 4);
    let mut glides: Vec<usize> = Vec::new();
    let mut folds: Vec<usize> = Vec::new();
    let mut take_next = 0.0;
    for (i, s) in segs_in.iter().enumerate() {
        let mut s2 = s.clone();
        s2.dur = Some(s.dur.unwrap_or(defaults[i]) - take_next);
        take_next = 0.0;
        segs.push(s2);
        if let Some(next) = segs_in.get(i + 1) {
            if !s.word_end && next.is_nucleus() {
                let (base, len) = if s.labial {
                    ("w", p.lab_glide)
                } else if palatal(s) && p.pal_glide > 0.0 {
                    ("j", p.pal_glide)
                } else if palatal(s) {
                    (if s.base == "c" { "ç" } else { "ʝ" }, p.pal_fric)
                } else {
                    ("", 0.0)
                };
                if len > 0.0 {
                    let g = len.min(0.4 * next.dur.unwrap_or(defaults[i + 1]));
                    if base == "w" {
                        glides.push(segs.len());
                    }
                    folds.push(segs.len());
                    segs.push(PieSeg { base: base.into(), dur: Some(g), ..Default::default() });
                    take_next = g;
                }
            }
        }
    }
    let mut plan = plan_core(&segs, &glides, question, o, p);
    for &k in folds.iter().rev() {
        let t1 = plan.segments[k].t1;
        plan.segments[k - 1].t1 = t1;
        plan.segments.remove(k);
    }
    plan
}

/// `glides`: indices of the [w] release transitions plan_pie_utterance inserted (tuned apart from a real /w/).
fn plan_core(segs_in: &[PieSeg], glides: &[usize], question: bool, o: &PlanOptions, p: &PieOptions) -> Plan {
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
            ipa: engine_ipa(s, p).to_string(),
            dur: Some(*d),
            stress: Some(if s.is_vowel() && s.accent { 1 } else { 0 }),
            f0: None,
            word_end: Some(s.word_end),
        })
        .collect();
    let po = PlanOptions { accent: Some(p.accent), intonation: Some(if question { Intonation::Question } else { Intonation::Statement }), ..o.clone() };
    let mut plan = plan_utterance(&segs, &po);
    let w_spans: Vec<(f64, f64)> = plan.segments.iter().map(|s| (s.t0, s.t1)).collect();
    let f0 = o.f0.unwrap_or(110.0);
    // plain voiceless stops; the tuned /i/ /j/ /w/
    for g in plan.gestures.iter_mut() {
        match g.label_str() {
            "voiceless" | "h" if g.targets.has(ControlKey::ABDUCTION) => {
                g.strength = p.devoice_str;
                g.t1 -= p.devoice_early_fric;
                if p.devoice_ramp_in > 0.0 {
                    g.ramp_in = p.devoice_ramp_in;
                }
            }
            "aspiration" => {
                let was_aspirated = g.targets.get(ControlKey::ABDUCTION).unwrap_or(0.0) > 0.3;
                g.targets = Targets::default().with(ControlKey::ABDUCTION, p.asp_ab);
                g.strength = p.devoice_str_stop;
                g.t1 -= p.devoice_early;
                if p.devoice_ramp_in > 0.0 {
                    g.ramp_in = p.devoice_ramp_in;
                }
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
            "a" | "ɑ" => {
                if let Some(j) = g.targets.get(ArticKey::Jaw) {
                    g.targets.set(ArticKey::Jaw, j + p.a_jaw);
                }
            }
            "u" => {
                if let Some(x) = g.targets.get(ArticKey::TbX) {
                    g.targets.set(ArticKey::TbX, x + p.u_back);
                }
                if let (Some(la), true) = (p.u_lip_ap, g.targets.has(ArticKey::LipAp)) {
                    g.targets.set(ArticKey::LipAp, la);
                }
            }
            "o" => {
                if let Some(x) = g.targets.get(ArticKey::TbX) {
                    g.targets.set(ArticKey::TbX, x + p.o_back);
                }
            }
            "e" => {
                if let Some(y) = g.targets.get(ArticKey::TbY) {
                    g.targets.set(ArticKey::TbY, y + p.e_raise);
                }
            }
            "s" if g.strength >= 40.0 => {
                if let Some(x) = g.targets.get(ArticKey::TtX) {
                    g.targets.set(ArticKey::TtX, x + p.s_tip_fwd);
                }
                if let Some(y) = g.targets.get(ArticKey::TtY) {
                    g.targets.set(ArticKey::TtY, y + p.s_tip_up);
                }
                if let Some(j) = p.s_jaw {
                    g.targets.set(ArticKey::Jaw, j);
                }
                if let Some(lp) = p.s_lip_prot {
                    g.targets.set(ArticKey::LipProt, lp);
                    g.targets.set(ArticKey::LipWidth, 1.0);
                }
            }
            "t" | "d" | "n" if p.coronal_front != 0.0 => {
                if let Some(x) = g.targets.get(ArticKey::TtX) {
                    g.targets.set(ArticKey::TtX, x + p.coronal_front);
                }
            }
            "l" if g.strength >= 40.0 && (p.l_tb_x.is_some() || p.l_tb_y.is_some()) => {
                if let Some(x) = p.l_tb_x {
                    g.targets.set(ArticKey::TbX, x);
                }
                if let Some(y) = p.l_tb_y {
                    g.targets.set(ArticKey::TbY, y);
                }
            }
            "c" | "ɟ" if g.strength >= 40.0 && p.c_front != 0.0 => {
                if let Some(x) = g.targets.get(ArticKey::TbX) {
                    g.targets.set(ArticKey::TbX, x + p.c_front);
                }
            }
            "j" => {
                if let Some(y) = g.targets.get(ArticKey::TbY) {
                    g.targets.set(ArticKey::TbY, y + p.j_raise);
                }
            }
            "w" => {
                if g.strength >= 40.0 {
                    let mid = 0.5 * (g.t0 + g.t1);
                    let glide = glides.iter().any(|&k| w_spans.get(k).is_some_and(|&(a, b)| mid > a - 0.04 && mid < b + 0.04));
                    if glide {
                        g.targets.set(ArticKey::LipAp, p.glide_lip_ap);
                    } else {
                        // a real /w/: rounder and further back than /u/, so it is not heard as a dark /l/
                        g.targets.set(ArticKey::LipAp, p.w_lip_ap);
                        g.targets.set(ArticKey::LipProt, p.w_lip_prot);
                        g.targets.set(ArticKey::TbX, p.w_tb_x);
                        if let Some(y) = g.targets.get(ArticKey::TbY) {
                            g.targets.set(ArticKey::TbY, y + p.w_tb_y);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    if p.vowel_open > 0.0 {
        let vspans: Vec<(f64, f64)> = (0..n)
            .filter(|&i| segs_in[i].is_vowel() && plan.segments[i].t1 - plan.segments[i].t0 < p.vowel_open_max)
            .map(|i| (plan.segments[i].t0, plan.segments[i].t1))
            .collect();
        for &(a, b) in &vspans {
            let m = 0.5 * (a + b);
            let h = (0.5 * p.vowel_open * (b - a)).max(0.006);
            // only obstruents' closing gestures: glides (the w release of kʷ), laterals, nasals and trills keep theirs
            const OBSTRUENT: [&str; 13] = ["p", "b", "t", "d", "k", "g", "c", "ɟ", "s", "z", "χ", "ʁ", "ç"];
            for g in plan.gestures.iter_mut().filter(|g| g.strength >= 40.0 && OBSTRUENT.contains(&g.label_str())) {
                let before = g.t0 < a && g.t1 < b && g.t1 > a - 0.15;
                let after = g.t1 > b && g.t0 > a - 0.02 && g.t0 < b + 0.15;
                if before && g.t1 + g.ramp_out > m - h {
                    // the consonant before: it lets go by the open stretch
                    if g.t1 > m - h - 0.004 {
                        g.t1 = (m - h - 0.004).max(g.t0);
                    }
                    g.ramp_out = (m - h - g.t1).max(0.004);
                } else if after && g.t0 - g.ramp_in < m + h {
                    // the consonant after: it starts closing only after the open stretch
                    if g.t0 < m + h + 0.004 {
                        g.t0 = (m + h + 0.004).min(g.t1);
                    }
                    g.ramp_in = (g.t0 - (m + h)).max(0.004);
                }
            }
        }
    }
    if p.velum_lead >= 0.0 {
        for g in plan.gestures.iter_mut().filter(|g| g.label_str() == "velum") {
            // the planner starts it 20 ms before the nasal with a 50 ms ramp
            let start = g.t0 + 0.02;
            g.ramp_in = p.velum_ramp;
            g.t0 = (start - p.velum_lead + p.velum_ramp).min(g.t1);
        }
    }
    if p.h_release >= 0.0 {
        const OBSTRUENT: [&str; 13] = ["p", "b", "t", "d", "k", "g", "c", "ɟ", "s", "z", "χ", "ʁ", "ç"];
        for i in (0..n).filter(|&i| segs_in[i].base == "h") {
            let (a, b) = (plan.segments[i].t0, plan.segments[i].t1);
            let r = a + p.h_release * (b - a);
            for g in plan.gestures.iter_mut().filter(|g| g.strength >= 40.0 && OBSTRUENT.contains(&g.label_str())) {
                if g.t0 < a && g.t1 <= b && g.t1 > a - 0.15 && g.t1 + g.ramp_out > r {
                    // the consonant before h lets go by the time the breath is on its own
                    if g.t1 > r - 0.004 {
                        g.t1 = (r - 0.004).max(g.t0);
                    }
                    g.ramp_out = (r - g.t1).max(0.004);
                }
            }
        }
    }
    for spec in &p.vowel_deltas {
        let parts: Vec<&str> = spec.split(':').collect();
        if parts.len() != 3 {
            continue;
        }
        let (Some(k), Ok(d)) = (ArticKey::from_name(parts[1]), parts[2].parse::<f64>()) else { continue };
        for g in plan.gestures.iter_mut().filter(|g| g.label_str() == parts[0]) {
            if let Some(x) = g.targets.get(k) {
                g.targets.set(k, x + d);
            }
        }
    }
    let ps = plan.segments.clone();
    // h₂ and h₃: the uvular constriction and, for χ, how open the glottis is
    for (i, s) in segs_in.iter().enumerate() {
        let (a, b) = (ps[i].t0, ps[i].t1);
        let over = |g: &Gesture| g.t1 > a - 0.02 && g.t0 < b + 0.02;
        if s.base == "χ" {
            for g in plan.gestures.iter_mut().filter(|g| over(g)) {
                if g.label_str() == "voiceless" {
                    g.targets.set(ControlKey::ABDUCTION, p.chi_abd);
                } else if g.label_str() == "χ" && g.strength >= 40.0 {
                    if let Some(y) = g.targets.get(ArticKey::TbY) {
                        g.targets.set(ArticKey::TbY, y + p.chi_raise);
                    }
                }
            }
        } else if s.base == "ʁ" && p.gh_raise != 0.0 {
            for g in plan.gestures.iter_mut().filter(|g| over(g) && g.label_str() == "ʁ" && g.strength >= 40.0) {
                if let Some(y) = g.targets.get(ArticKey::TbY) {
                    g.targets.set(ArticKey::TbY, y + p.gh_raise);
                }
            }
        }
        // syllabic r̥ as a vocoid with tip contacts across its middle
        if s.base == "r" && s.syllabic && p.syl_r_trill > 0 {
            let k_n = p.syl_r_trill as usize;
            let (c_a, c_b) = (a + 0.2 * (b - a), a + 0.8 * (b - a));
            let per = (c_b - c_a) / k_n as f64;
            for k in 0..k_n {
                let c0 = c_a + k as f64 * per;
                let close = Targets::default().with(ArticKey::TtX, 6.25).with(ArticKey::TtY, p.trill_close_y);
                let open = Targets::default().with(ArticKey::TtX, 6.1).with(ArticKey::TtY, p.trill_open_y);
                plan.gestures.push(Gesture::new(c0 - 0.006, c0 + p.trill_close * per, 0.004, 0.004, 40.0, close).speed(p.trill_speed).label("r"));
                plan.gestures.push(Gesture::new(c0 + p.trill_close * per, c0 + per - 0.006, 0.004, 0.004, 40.0, open).speed(p.trill_speed).label("r"));
            }
        }
    }
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
                    let close = Targets::default().with(ArticKey::TtX, 6.25).with(ArticKey::TtY, p.trill_close_y);
                    plan.gestures.push(Gesture::new(c0 - 0.006, c0 + p.trill_close * per, 0.004, 0.004, 40.0, close).speed(p.trill_speed).label("r"));
                    if k + 1 < k_n {
                        let open = Targets::default().with(ArticKey::TtX, 6.1).with(ArticKey::TtY, p.trill_open_y);
                        plan.gestures.push(Gesture::new(c0 + p.trill_close * per, c0 + per - 0.006, 0.004, 0.004, 40.0, open).speed(p.trill_speed).label("r"));
                    }
                }
            }
        }
    }
    if p.r_tb_x.is_some() || p.r_tb_y.is_some() {
        for i in (0..n).filter(|&i| segs_in[i].base == "r") {
            let mut tg = Targets::default();
            if let Some(x) = p.r_tb_x {
                tg.set(ArticKey::TbX, x);
            }
            if let Some(y) = p.r_tb_y {
                tg.set(ArticKey::TbY, y);
            }
            plan.gestures.push(Gesture::new(ps[i].t0, ps[i].t1, 0.02, 0.02, p.r_tb_str, tg).label("r body"));
        }
    }
    for (i, s) in segs_in.iter().enumerate() {
        let (t0, t1) = (ps[i].t0, ps[i].t1);
        if s.labial {
            let tg = Targets::default().with(ArticKey::LipAp, p.lab_ap).with(ArticKey::LipProt, p.lab_prot).with(ArticKey::LipWidth, 0.065);
            plan.gestures.push(Gesture::new(t0 - p.lab_lead, t1 + p.lab_carry, 0.03, 0.03, p.lab_str, tg).label("ʷ"));
        }
        let colour = match s.base.as_str() {
            "h" => p.h_asp_lowpass,
            "χ" => p.chi_asp_lowpass,
            _ => 0.0,
        };
        if colour > 0.0 {
            plan.gestures.push(Gesture::new(t0 - 0.02, t1 + 0.015, 0.015, 0.015, 4.0, Targets::default().with(ControlKey::ASP_LOWPASS, colour)).label("breath colour"));
        }
        if is_voiced_stop(&s.base) {
            let mut tg = Targets::default();
            if let Some(y) = p.vstop_larynx {
                tg.set(ArticKey::Larynx, y);
            }
            if let Some(r) = p.vstop_root {
                tg.set(ArticKey::Root, r);
            }
            if !tg.is_empty() {
                plan.gestures.push(Gesture::new(t0 - 0.01, t1 - 0.005, 0.02, 0.02, p.vstop_str, tg).label("voice bar"));
            }
        }
        if s.breathy {
            let mut tg = Targets::default().with(ControlKey::ABDUCTION, p.mur_ab);
            if p.mur_level > 0.0 {
                tg.set(ControlKey::MURMUR, p.mur_level);
            }
            plan.gestures.push(Gesture::new(t1 - 0.01, t1 + p.mur_dur, 0.015, 0.03, 6.0, tg).label("ʱ"));
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

/// Set a voice up for PIE: the source the defaults were tuned with (a clean, slightly lax voice)
/// and the dorsal frication source. Call before scheduling a PIE plan.
pub fn setup_voice(v: &mut crate::voice::human_voice::HumanVoice, f0: f64, p: &PieOptions) {
    use crate::voice::human_voice::SourcePatch;
    v.set_source(&SourcePatch { f0: Some(f0), breathiness: Some(0.05), tension: Some(0.4), ..Default::default() });
    v.airway.noise_gain.dorsal = p.dorsal_noise;
    v.airway.noise_gain.teeth_lowpass = p.sibilant_lowpass;
    v.airway.noise_gain.dorsal_re = p.dorsal_re;
    v.airway.noise_gain.dorsal_offset = p.dorsal_offset;
    v.airway.noise_gain.dorsal_offset2 = p.dorsal_offset2;
    v.airway.noise_gain.dorsal_split = p.dorsal_split;
    v.airway.noise_gain.dorsal_area = p.dorsal_area;
    v.airway.noise_gain.dorsal_lowpass = p.dorsal_lowpass;
    v.airway.noise_gain.dorsal_steep = p.dorsal_steep;
    v.airway.noise_gain.glottis_lowpass = p.glottis_lowpass;
    v.airway.noise_gain.glottis *= p.glottis_noise;
    v.airway.noise_gain.constr_area = p.constr_area;
    v.airway.noise_gain.burst_lowpass = p.burst_lowpass;
    v.airway.noise_gain.burst_gain = p.burst_gain;
    v.airway.noise_gain.burst_time = p.burst_time;
    v.airway.noise_gain.burst_gain_lab = p.burst_gain_lab;
    v.airway.noise_gain.burst_gain_front = p.burst_gain_front;
    v.murmur_chink = p.mur_chink;
    v.murmur_oq = p.mur_oq;
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
        assert_eq!(labels("*h₂éwsōs"), ["χ", "é", "w", "s", "oː", "s"]);
        assert_eq!(labels("*ǵʰéyōm"), ["ɟʱ", "é", "j", "oː", "m"]);
        assert_eq!(labels("gʷʰén-ti"), ["gʷʱ", "é", "n", "t", "i"]);
        assert_eq!(labels("h₃ékʷ"), ["ʁʷ", "é", "kʷ"]);
        assert_eq!(labels("ḱm̥tóm"), ["c", "m̩", "t", "ó", "m"]);
        assert_eq!(labels("dʰugh₂tḗr"), ["dʱ", "u", "g", "χ", "t", "éː", "r"]);
        assert_eq!(labels("*léukā déiwos"), ["l", "é", "w", "k", "aː", "d", "é", "j", "w", "o", "s"]);
    }

    #[test]
    fn reads_corpus_ipa() {
        assert_eq!(labels("χáw.sos"), ["χ", "á", "w", "s", "o", "s"]);
        assert_eq!(labels("ń̩dʱ.ɡʷʱi.tom"), ["ń̩", "dʱ", "gʷʱ", "i", "t", "o", "m"]);
        assert_eq!(labels("ɣʷkʷbʱós"), ["ʁʷ", "kʷ", "bʱ", "ó", "s"]);
        assert_eq!(labels("wih.róːs"), ["w", "i", "h", "r", "óː", "s"]);
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
