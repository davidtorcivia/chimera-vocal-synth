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
| 2026-11-08 | dark l: l gets its own tongue body (back, lowered; it had none) — l 48 → 8 %, l̩ 59 → 35 % | 27.7 % | 27.6 % | |

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
