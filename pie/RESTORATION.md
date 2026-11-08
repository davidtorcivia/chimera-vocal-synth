# Restoration v2: plan

Corpus v2 (2026-11-08): Chimera renders (dcc37da) converted by ElevenLabs into Zazy and Valeria (all clips) and
Antonio and W02 (3-hour random sets). The conversions are time-aligned with their sources (envelope lag 0 ms
median, all clips within ±10 ms, no drift), so every phone span of the Chimera timeline holds in the converted
audio. Restoration edits the converted audio inside those spans, in the target voice; it never pastes Chimera in.

## What "correct" means (owner, 2026-11-08)
- Sounds the recognizer can hear: PhoneticXeus per-phone hits on their spans (pie/gop.py, broad).
- Sounds it cannot: published phonetic targets — murmur breathiness after bʰ dʰ gʰ as measured on real
  breathy-voiced speakers (H1*-H2*, CPP), trills with 2-3 contacts at ~25-30 Hz, h with real aspiration energy.
- Never the Chimera source itself as the target.

## Damage map, Zazy (all 2,119 male clips; source → converted, broad misses)
Converter helps: χ 53 → 23 %, d 34 → 22, ə 82 → 71, t 26 → 18, r̩ 38 → 30; h 68 → 65.
Converter hurts: kʷ 23 → 44 (heard k: rounding lost), k 15 → 31, p 20 → 28 (heard b), j 38 → 45, w 34 → 42,
r 29 → 36, e 14 → 20, o 5 → 10, s 10 → 14. Blind spots (both): c, ɟ, bʱ, dʱ, gʱ.

## Damage map, all four voices (full sets; source → converted broad misses)
Whole sets: Zazy 26.9 → 28.4 %, Antonio 26.9 → 28.6, Valeria 29.5 → 30.5, W02 29.5 → 31.6 (conversion costs only
1-2 points overall). Same clips: Zazy 28.5 vs Antonio 28.6 (1,210 male; a tie); Valeria 30.4 vs W02 31.6 (1,212
female; Valeria better on 642 clips, worse on 429).
Consistent losses in every voice: kʷ +16-22 points, k +15-19, j +7-15, w +7-16, r +7-11, m +4-7, p +3-8;
female only: s +12-16 (11 → 23-27 %), u +12-17 (23 → 35-40 %), n +4-5. Consistent gains: χ −22 to −30, d −10 to −13,
t −6 to −10, ə −8 to −12, h −3 to −5, gʷ (female) −8 to −12.

## Acoustic map (Phase 0, `pie/acoustic_map.py`, 300 clips per voice, source → converted; `pie/cover/acoustic-map.json`)
Survives: male murmur (H1*-H2* over modal +7 → +8/+9 dB), h (level kept, brightness 0.2-0.4 → 0.43-0.64),
χ (improved), Zazy's trill (2+ contacts 56 → 65 %), VOT (10-20 ms).
Damaged: stop bursts 3-8 dB weaker in every voice (→ k p kʷ misheard); female s slides toward ʃ (centroid 6.0 kHz
Valeria, 4.9 W02; heard ʃ 1,435× in Valeria); kʷ rounding transition halved (F2 rise 700-970 → 380-800 Hz);
trill flattened in Antonio, Valeria, W02 (no contact in ~40 % of r); female murmur weak in the source (+2 dB) and
lost in Valeria (−2); e F2 falls 50-190 Hz. LTAS: W02 thin below 500 Hz, Antonio dull at 6-8 kHz.
QC voicing: two-thirds of Zazy's flags are creaky-but-voiced vowels (meter too strict), a third whispered short ə.
Order: bursts (EQ), female s (EQ), kʷ transition (resynthesis), trill and female murmur, e. Engine note: the
female voice's murmur is too weak (+2 dB vs +7 male).

## Phases
0. Measure: the damage map for all four voices, plus acoustic measures per span — murmur H1*-H2*/CPP, trill
   contacts, h and χ band energy, VOT and burst energy for p t k, F2 transitions for kʷ gʷ w j, vowel formants,
   per-voice long-term spectrum. Find out why 798 Zazy clips fail the QC voicing check (meter or voice).
1. Tools, all in the target voice:
   - EQ first: per-voice static EQ to a target tonal balance; span-local band EQ with smooth ramps (2-8 kHz for
     h s χ, burst presence for p t k, resonance trims for drifted vowels, de-essing where s slid to ʃ).
   - Resynthesis where EQ cannot reach (vocoder analysis: f0, envelope, aperiodicity; edit inside the span;
     crossfade): rounding transitions (kʷ gʷ w), glide targets (j w), trill contact rhythm, murmur breathiness,
     h breath shaped by the following vowel's own formants.
2. Build each repair alone, in order of what it can gain (kʷ/k; glides, trill, p; e/o; murmur; h). Each must improve
   its target, not hurt its neighbours, keep naturalness (UTMOS) and leave no splice artefacts, and pass a blind
   A/B on the listening page. EQ-only versions are tried first.
3. Run the chain on all 6,620 clips; rebuild the QC gate for this corpus (floors from measured clean clips, a
   voicing meter that works on low male voices); export the Kokoro set (24 kHz, train/val lists, four speakers).
4. The Chimera → natural converter (mel-to-mel + vocoder), trained on the restored clips paired with their sources.

## Pilot, round 1 (2026-11-08; `pie/restore2.py`, `pie/restore_pilot.py`, `pie/pilot_score.py`; 12 clips per voice)
- WORLD-vocoder repairs fail: an unedited WORLD round trip alone costs naturalness (UTMOS −0.04 to −0.09) and
  accuracy (+0.3 to +2.3 points); kʷ stretch +1.5 to +4.4 points worse; murmur UTMOS −0.21 to −0.36 (female).
  Uncorrected H1-H2 shows the murmur edit does raise breathiness (10.4 → 17.7 dB); the Iseli-Alwan H1*-H2* hides it
  (an H1 boost drags F1 onto H1). Next round: STFT-domain gain edits only, nothing resynthesized.
- Trill dips (no vocoder): accuracy −0.9 to 0 points (better), UTMOS −0.07 to −0.15: keep, make gentler.
- EQ: female s correction works (W02 s misses 10 → 4 of 43, centroid 5.3 → 7.2 kHz); burst boost raises bursts
  3-6 dB with no recognition gain; tonal EQ neutral for UTMOS and accuracy. Owner: EQ is for mastering, judged by
  ear and naturalness; one repair at a time. Tonal EQ: boosts only (the reference voices are thinner than ours).
Step 1 listening page (mastering EQ: conversion / half / full, loudness-matched):
https://claude.ai/artifact/G339o7pt6nqvPC3By29QAv

Step 1 decided (owner): full mastering EQ for all four voices.
Step 2 (female s, on top of step 1; cut 2.5-5 kHz / lift 6.5-10 kHz inside s): Valeria s misses 4 → 3 → 3 → 2 of 42
(centroid 6.2 → 6.9 / 7.0 / 7.2 kHz) for cut/lift 4/3, 5/4, 7/6 dB; W02 10 → 7 → 6 → 5 of 43 (5.6 → 7.0 / 7.2 / 7.4 kHz)
for 5/4, 7/6, 9/8 dB; UTMOS unchanged (±0.02). On the lab page for the owner's choice.

Step 2 decided (owner): strong s (Valeria 7/6 dB, W02 9/8 dB).
Step 3 (bursts, on steps 1-2): fixed 2-8 kHz lift or per-burst fill of the shortfall against the source (contrast
transfer, `restore2.contrast_transfer`, STFT gain with original phase). Bursts rise up to ~6 dB but p t k are not
recognized better; misses Zazy 30.7 → 31.6/30.5, Antonio 31.5 → 33.3/33.5, Valeria 32.7 → 32.1/31.4, W02 35.0 →
36.1/35.6 (fixed / half fill); UTMOS ±0.02. Recommendation: skip (or Valeria only); on the page for the owner's ear.
Research (agent, 2026-11-08): Hazan & Simpson 1996/1998 cue enhancement (+12 dB bursts, ramped, abrupt steps hurt);
LPC-domain formant edits keeping the residual (Kuwabara & Takagi 1987) for kʷ and e; breathiness needs
pitch-synchronous, LPC-shaped aspiration noise (Klatt & Klatt 1990; Hermes 1991) — an H1 boost alone reads as
nasality in a high voice; trills: transfer the source's 500-3,500 Hz envelope only, leave <500 Hz (voicing) intact
(a 2026 IberSPEECH paper cited, not verified); pre-distortion literature is thin.

Step 3 decided (owner): half fill (contrast transfer, alpha 0.5) for all four voices.
Step 4 (trill, on steps 1-3) — no repair kept. Envelope transfer from the source above 500 Hz (`fix_trill_env`): UTMOS
−0.06 to −0.17, r unchanged; softer (−12 dB floor) still −0.04 to −0.14; synthetic dips (8 dB) −0.05 to −0.13, r
unchanged. A mid-band closure detector (built from an unverified paper) found almost no contacts even in the Chimera source;
the owner hears trills, so that detector is miscalibrated for these voices — the conclusion that the engine has
no trill was WRONG and is withdrawn. On the full-band contact count, which matches listening, the source has 2+
contacts in 56-59 % of trills.
The old corpus's re-trill (PIE repo tts/vtl/retrill.py → pie_vtl.patch_trill, reused as is): contacts cut to
−20 dB re the next vowel, 40 % of each 40 ms cycle, 5 ms cosine edges, only on r the timeline marks as trills.
2+ contacts Zazy 57 → 70 %, Antonio 17 → 43, Valeria 26 → 35, W02 42 → 46 %; UTMOS +0.00 / −0.08 / −0.03 / −0.04;
r recognition unchanged. On the lab page for the owner.
