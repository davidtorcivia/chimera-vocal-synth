#!/usr/bin/env python3
"""Restoration v2 pilot: 12 clips per voice in three versions — conv (as converted), eq (EQ only), all (EQ + the
resynthesis repairs) — under full/restore-pilot/<version>/<TAG>/. Per-voice settings come from the Phase-0 damage
map (pie/cover/acoustic-map.json); the per-voice EQ curve from 40 of the voice's own clips.

    python pie/restore_pilot.py
"""
import json, random, sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import restore2 as R

FULL = Path.home() / "AI/chimera-pie-data/full"
OUT = FULL / "restore-pilot"
SET = {
    #           gender   burst dB  s cut/lift   kw    trill  murmur
    "ZAZY":    ("male",   5.0,     (0, 0),      True, False, False),
    "ANTONIO": ("male",   6.0,     (0, 0),      True, True,  False),
    "VALERIA": ("female", 5.0,     (5.0, 4.0),  True, True,  True),
    "W02":     ("female", 3.0,     (7.0, 6.0),  True, True,  True),
}


VARIANTS = ["conv", "tone", "bursts", "s", "eq", "all"]


def main():
    clips = json.load(open(FULL / "restore-pilot-clips.json"))
    curves = {}
    for tag, (gender, burst, (cut, lift), kw, trill, mur) in SET.items():
        stems = sorted(p.stem for p in (FULL / "eleven-v2" / tag).glob("*.wav"))
        random.Random("eq-" + tag).shuffle(stems)
        xs = [sf.read(FULL / "eleven-v2" / tag / f"{s}.wav", dtype="float64")[0] for s in stems[:40]]
        edges, g = R.voice_curve(xs, 44100, gender)
        curves[tag] = {"edges": edges.tolist(), "gains_db": [round(float(v), 2) for v in g]}
        for v in VARIANTS:
            (OUT / v / tag).mkdir(parents=True, exist_ok=True)
        for s in clips[tag]:
            x, sr = sf.read(FULL / "eleven-v2" / tag / f"{s}.wav", dtype="float64")
            src, _ = sf.read(FULL / "chimera-v2/renders" / f"{s}.wav", dtype="float64")
            ph = R.phones(FULL / "timing" / f"{s}.json")
            peak = np.max(np.abs(x))
            out = {"conv": x,
                   "tone": R.eq_voice(x, sr, edges, g),
                   "bursts": R.eq_bursts(x, sr, ph, burst),
                   "s": R.eq_s(x, sr, ph, cut, lift)}
            e = R.eq_s(R.eq_bursts(x, sr, ph, burst), sr, ph, cut, lift)     # the EQ set: tone left out (it hurt)
            out["eq"] = e
            a = e
            if kw:
                a = R.fix_kw(a, sr, ph)
            if trill:
                a = R.fix_trill(a, sr, ph, src)
            if mur:
                a = R.fix_murmur(a, sr, ph)
            out["all"] = a
            for v, y in out.items():
                y = y / max(1e-9, np.max(np.abs(y))) * min(0.95, peak)          # same peak as the conversion
                sf.write(OUT / v / tag / f"{s}.wav", y, sr, subtype="PCM_16")
        print(tag, "tone EQ (dB per third octave from 100 Hz):", " ".join(f"{v:+.1f}" for v in g), flush=True)
    json.dump(curves, open(OUT / "eq-curves.json", "w"), indent=1)


if __name__ == "__main__":
    main()
