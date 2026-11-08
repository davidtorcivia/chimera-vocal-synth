# PIE on Chimera: results log

Scores are PhoneticXeus phone misses on known spans (`pie/gop.py`, broad: near neighbours that do not change the PIE
phoneme count as right), unless marked otherwise. Dev and held-out test are halves of 326 corpus sentences
(pie_vtl v10.1 timelines); the 24-sentence pilot is a held-out subset sent through ElevenLabs speech-to-speech
(eleven_multilingual_sts_v2, stability 0.9, similarity 0, the corpus voices).

| Date | Engine | Dev | Held-out | Pilot raw → converted |
|---|---|---|---|---|
| 2026-11-07 | VocalTractLab v10 (the current corpus) | | 54.3 % | 55.3 → 56.5 % |
| 2026-11-07 | Chimera, first pilot (fe2c125) | | 33.2 % | 35.3 → 38.9 % |
| 2026-11-07 | Chimera, voice onset + vowel guard + stops (29397c1) | 29.7 % | 30.2 % | 32.4 → 35.6 % |
| 2026-11-07 | χ back to the first pilot's single-point, brighter source (G1) | 30.0 % | 30.2 % | χ/s set: 27.7 → 36.0 % became 26.4 → 32.6 % (two conversions) |
| 2026-11-08 | trill gets 110 ms (from its neighbours) instead of 90: r 44 → 27 % missed | 29.4 % | 29.7 % | |
| 2026-11-08 | velum lowers 20 ms before a nasal over 20 ms (was 70 ms, English-like) | 28.6 % | 29.3 % | |
| 2026-11-08 | the [w] glide after kʷ gʷ 10 ms instead of 30 (kʷ 24 → 13 % on a kʷe set) | 28.3 % | 28.4 % | |
| 2026-11-08 | dark l: l gets its own tongue body (back, lowered; it had none) — l 48 → 8 %, l̩ 59 → 35 % | 27.7 % | 27.6 % | 27.4 → 33.8 % (pilot 3, 1,110 credits) |
| 2026-11-08 | u: tongue less retracted, tighter lips — it nearly closed on the velum (0.01 cm², voicing lost, 25 dB under a in pápu); weak u 27 → 7 % | 27.4 % | 27.5 % | |
| 2026-11-08 | u midpoint (u_back −0.2, lips 0.16): male F2 back to +28 Hz of reference (the first fix fronted it +132); weak u 10 %; the female u stays ~150 Hz front because backing it re-closes the velar passage | 27.3 % | 27.0 % | |

Conversion tests (`pie/convtest.py`, 12 dev sentences rich in χ and s, 678 credits each): today's χ converted 36.0 and
36.5 % on two runs (the converter's run-to-run spread is about half a point); the pilot's χ source alone 31.7 %
(χ 58 → 50 % missed, s 41 → 19 %). Raw scores barely move, so χ is now judged after conversion.

Human floor (`pie/humanfloor.py`): PhoneticXeus on 1,000 real recorded words from VoxAngeles (95 languages,
audited IPA), scored like the hymn's whole-transcription measure, misses 44.9 % of the sounds. The hymn from
Chimera scores 37-43 % raw on that measure, so whole-transcription numbers are at the recognizer's floor; the
per-phone span score (30 %) still has room.

Voice quality (`pie/voicequal.py`, dawn hymn): male jitter 1.2 %, shimmer 9 %, HNR 16 dB (human range); female
jitter 0.8 %, HNR 21 dB, 4-8 kHz band 28 dB under 0-4 kHz (16 dB after conversion): too steady and too dark.

Dawn hymn, whole-hymn transcription vs its 189 target sounds (`pie/wordscore.py` logic; the site's Kokoro voice
has its own timing): male VTL 40 / converted 52, Chimera 37 / converted 41, site Kokoro 54 %; female VTL 57 / 64,
Chimera 43 / 46, Kokoro 59 %.

What the recognizer cannot hear in any source: murmured stops, palatal stops, ʁʷ. Checked by acoustics instead
(`pie/murmur.py` H1*-H2* +5 to +6.5 dB; palatal F2 transitions; `pie/glides.py` trill contacts; `pie/vot.py`
voice onset 14-20 ms; `pie/vowels.py` formants within ~25 Hz of reference; `pie/accent.py` accent lift +3.4 st).

Open: χ survives conversion worse than at the first pilot (raw 30 %, converted 63 % vs 33 %); s heard as z;
n, m̩, n̩ and kʷ lost a little in the last round; h before consonants (inaudible in any language).

The listening page is built by `pie/page/build_page.py` from `pie/page/pilot.json` and audio in the data folder.

Recognizer on real speech (`pie/humanrecall.py`, VoxAngeles words containing the sound): c 5 %, ɟ 8 % heard (true
blind spots), ə 17 %, ʁ 28 %, kʷ 34 %, χ 48 %, j 50 %, w 57 %, h 65 %, r 89 %; murmured stops 2 of 9 Kannada and
Konkani words. Chimera's j, w and ə are already at or above the human rate; r was far below (61 %) and is the
reason for the longer trill. Converter noise: the same render converted twice differs by ~2 points on 12
sentences, so conversion tests only catch large effects.

Tried and not kept (2026-11-08): releasing an obstruent before h early (h +2, s -2 points); a closer o (-1);
female-only e/i/coronal settings (within noise; the female gap of ~3 points is the recognizer's, VTL shows it too);
weaker voice bar for d (d before a consonant is heard as n; no setting helped, and it is 0.6 % of misses); a
stronger or longer murmur (heard less, not more). An apparent leak in t/d closures was the debug print, not the
airway: mouth pressure equals lung pressure and nothing flows. Tools: `piedebug` PIE_GESTURES=t0,t1 lists active
gestures; the `pie` example takes f.key=value / m.key=value for one voice only.

Dawn hymn, night engine (whole-transcription): male raw 33 / converted 39 %, female 43 / 44 % (evening 37 / 41,
43 / 46). Pilot 3 after conversion per sound vs pilot 2: χ 63 → 40, s 38 → 28, o 19 → 5, t 36 → 24, d 40 → 20 %
better; r 50 → 50 (the converter flattens the better trill), m 13 → 32, i 25 → 42, u 29 → 36, w 44 → 53 % worse
(14-40 instances each).

Full corpus render (`pie/fullrender.py`, 2026-11-08, engine b5b60d7): all 4,238 clips × voice of the VTL build on
their VTL timings with the build's per-clip pitch (±2 st) and accent (3.6-4.4 st); 10.4 h, 0 errors, 0 length
mismatches, about 16 min on the Mac. The corpus QC gate (`pie/fullqc.py`): h/χ brightness floors are for restored
audio and fail raw (expected); 616 clips under 97 % voiced (520 male), 1,040 with more than one weak vowel —
u (2,157 flags), i (806), o (525). The u defect is fixed above; voicing next.

Full corpus re-rendered at 6f1aa07 (u fix): voicing failures 616 → 362, weak-vowel failures 1,040 → 540; without
the post-restore h/χ floors 3,361 / 4,238 pass. Remaining voicing flags are mostly trilled r̩ (voiced, but the
QC pitch tracker loses it between contacts). Hymn with the u fix (646 credits): ú of dʰúgh₂ter −14/−16 → −5/−6 dB
vs the median vowel; converted span score 35.7 → 38.9 %, the difference in e, w, t (untouched by the fix): the
voice changer's run-to-run spread on one passage. Credits overnight: 8,430 of the 12,000 approved.

Pilot 4 (final engine dcc37da, 1,110 credits): raw 27.9 %, converted 35.0 % (pilot 3: 27.4 / 33.8). Raw renders
per sentence nearly identical to pilot 3, converted swings ±2-4 sounds per sentence both ways: converter spread
on 24 sentences is about ±1.5 overall and ±4 per voice. VTL's raw renders pass the h/χ brightness floors (h share
median 0.84, χ 0.57 on the 326 local clips); Chimera's do not (h 0.19-0.37, χ 0.48). A 2 kHz breath band for h
reaches 0.52 with the same h recognition but costs 0.4 points on neighbouring vowels: left off, owner to decide.
Credits overnight: 9,540 of 12,000.

Final full render (dcc37da, ~/AI/chimera-pie-data/full/chimera-v2, 4,238 clips, 0 errors): gate without the
post-restore h/χ floors 3,225 / 4,238 pass (first u fix 3,361; before it about 2,700): voicing 408, weak vowels 665
(the midpoint u trades a little loudness for a back u). Earlier renders kept as chimera-v2-b5b60d7 and -6f1aa07.

Voice test (2026-11-08, `pie/voicetest.py`, passage in `pie/cover/`: 41 corpus words, every sound ≥ 4×, 24.7 s;
5,083 credits). Converted misses: current JD3 33.3 / 40.4 (two runs), W02 42.1 / 39.9; candidates male 32.5-39.9,
female 36.8-39.5 — all inside the repeat spread. Murmur 17-19 / 21 missed in every voice. Converted noise floor:
hr-Ivana −64, es-Sandra −62, W02 −46 dB; cs-Hanka 8.9 kHz and el-Jane 12.4 kHz band (dull); de-Tristan −30 dB
(noisy). Best-of-two conversions: 34.4 → 31.1 % when chosen and scored on the same sounds, ±0.9 when chosen on
half and scored on the other half — no real gain.
