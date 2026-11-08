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
