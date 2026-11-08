#!/usr/bin/env python3
"""Vowel quality: F1/F2 (Praat Burg) at the middle of every vowel of a run, by vowel and body, against reference
targets (Hillenbrand et al. 1995 adult means for [i ɛ/e ɑ o u]; [e] between Hillenbrand's /eɪ/ onset and /ɪ/;
Peterson & Barney for [ə]). Also F1/F2 at 25% and 75% to show undershoot.

    python pie/vowels.py NAME:WAVDIR:OFFSET [...] [--set dev] [--limit 80]
"""
import argparse, sys
from pathlib import Path

import numpy as np
import parselmouth

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval as E

REF = {  # (F1, F2) Hz
    "male": {"i": (342, 2322), "e": (440, 2050), "a": (750, 1350), "o": (497, 910), "u": (378, 997), "ə": (500, 1450)},
    "female": {"i": (437, 2761), "e": (520, 2450), "a": (920, 1550), "o": (555, 1035), "u": (459, 1105), "ə": (590, 1700)},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--set", default="dev")
    ap.add_argument("--limit", type=int, default=80)
    a = ap.parse_args()
    files = E.sets(a.set)
    files = [f for f in files if "JD3" in f][: a.limit // 2] + [f for f in files if "W02" in f][: a.limit // 2]
    for spec in a.specs:
        name, wdir, off = spec.split(":")
        off = float(off)
        acc = {}
        for f in files:
            stem = Path(f).stem
            wav = Path(wdir) / f"{stem}.wav"
            if not wav.exists():
                continue
            body = "female" if "W02" in stem else "male"
            snd = parselmouth.Sound(str(wav))
            fm = snd.to_formant_burg(time_step=0.005, max_number_of_formants=5, maximum_formant=5000 if body == "male" else 5500)
            for ut in E.utterances(f)["utterances"]:
                t = ut["start"]
                for s in ut["segs"]:
                    b = E.base(s["pie"])
                    if b in REF[body] and s["dur"] >= 0.07:
                        v = []
                        for q in (0.25, 0.5, 0.75):
                            tt = t + off + q * s["dur"]
                            v.append((fm.get_value_at_time(1, tt), fm.get_value_at_time(2, tt)))
                        if all(x == x for p in v for x in p):
                            acc.setdefault((body, b), []).append(v)
                    t += s["dur"]
        print(f"== {name}")
        for body in ("male", "female"):
            for b, (r1, r2) in REF[body].items():
                L = acc.get((body, b))
                if not L:
                    continue
                L = np.array(L)
                m = np.median(L[:, 1], 0)
                print(f"  {body:6} {b}  n={len(L):3}  mid F1 {m[0]:4.0f} F2 {m[1]:4.0f}   (ref {r1}/{r2}; ΔF1 {m[0]-r1:+4.0f} ΔF2 {m[1]-r2:+5.0f})   "
                      f"edges F1 {np.median(L[:,0,0]):4.0f}/{np.median(L[:,2,0]):4.0f} F2 {np.median(L[:,0,1]):4.0f}/{np.median(L[:,2,1]):4.0f}")


if __name__ == "__main__":
    main()
