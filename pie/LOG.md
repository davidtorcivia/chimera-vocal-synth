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

Dawn hymn, whole-hymn transcription vs its 189 target sounds (`pie/wordscore.py` logic; the site's Kokoro voice
has its own timing): male VTL 40 / converted 52, Chimera 37 / converted 41, site Kokoro 54 %; female VTL 57 / 64,
Chimera 43 / 46, Kokoro 59 %.

What the recognizer cannot hear in any source: murmured stops, palatal stops, ʁʷ. Checked by acoustics instead
(`pie/murmur.py` H1*-H2* +5 to +6.5 dB; palatal F2 transitions; `pie/glides.py` trill contacts; `pie/vot.py`
voice onset 14-20 ms; `pie/vowels.py` formants within ~25 Hz of reference; `pie/accent.py` accent lift +3.4 st).

Open: χ survives conversion worse than at the first pilot (raw 30 %, converted 63 % vs 33 %); s heard as z;
n, m̩, n̩ and kʷ lost a little in the last round; h before consonants (inaudible in any language).

The listening page is built by `pie/page/build_page.py` from `pie/page/pilot.json` and audio in the data folder.
