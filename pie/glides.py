#!/usr/bin/env python3
"""Glides, laterals and trills by acoustics, per run.
  w j l: F1 F2 F3 (Praat Burg) at the segment middle (glides are short: the middle 30%)
  r (non-syllabic): contacts = amplitude dips of >= 6 dB within the span (+10 ms), and their rate
References (adult male, approx.): [w] F1 ~300 F2 ~650-800 F3 ~2200-2400; [j] F1 ~280 F2 ~2100-2300 F3 ~2800+;
[l] (clear) F2 ~1200-1500 F3 ~2600+; trill 2-3 contacts at 20-35 Hz.

    python pie/glides.py NAME:WAVDIR:OFFSET [...] [--set dev] [--limit 80]
"""
import argparse, sys
from pathlib import Path

import numpy as np
import parselmouth

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval as E


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
        fmt, contacts = {}, []
        for f in files:
            stem = Path(f).stem
            wav = Path(wdir) / f"{stem}.wav"
            if not wav.exists():
                continue
            body = "female" if "W02" in stem else "male"
            snd = parselmouth.Sound(str(wav))
            fm = snd.to_formant_burg(time_step=0.004, max_number_of_formants=5, maximum_formant=5000 if body == "male" else 5500)
            inten = snd.to_intensity(minimum_pitch=400, time_step=0.001)  # an 8 ms window: trill contacts last ~10-20 ms
            for ut in E.utterances(f)["utterances"]:
                t = ut["start"]
                for s in ut["segs"]:
                    b = E.base(s["pie"])
                    if b in ("w", "j", "l") and not s["syllabic"]:
                        vals = []
                        for q in (0.35, 0.5, 0.65):
                            tt = t + off + q * s["dur"]
                            vals.append([fm.get_value_at_time(k, tt) for k in (1, 2, 3)])
                        v = np.nanmedian(np.array(vals, float), 0)
                        if np.all(np.isfinite(v)):
                            fmt.setdefault((body, b), []).append(v)
                    if b == "r" and not s["syllabic"]:
                        ts = np.arange(t + off - 0.04, t + off + s["dur"] + 0.04, 0.001)  # Chimera lengthens r into its neighbours
                        db = np.array([inten.get_value(x) for x in ts], float)
                        db = db[np.isfinite(db)]
                        if len(db) > 5:
                            # local minima at least 6 dB below the running max on both sides
                            n = 0
                            for i in range(2, len(db) - 2):
                                if db[i] == db[max(0, i - 8):i + 9].min() and db[:i].max() - db[i] >= 6 and db[i:].max() - db[i] >= 6:
                                    n += 1
                            contacts.append(n)
                    t += s["dur"]
        print(f"== {name}")
        for (body, b), L in sorted(fmt.items()):
            L = np.array(L)
            print(f"  {body:6} {b}  n={len(L):3}  F1 {np.median(L[:,0]):4.0f}  F2 {np.median(L[:,1]):4.0f}  F3 {np.median(L[:,2]):4.0f}")
        c = np.array(contacts)
        if len(c):
            print(f"  r  n={len(c)}  contacts: 0 {np.mean(c==0)*100:.0f}%  1 {np.mean(c==1)*100:.0f}%  2+ {np.mean(c>=2)*100:.0f}%  median {np.median(c):.0f}")


if __name__ == "__main__":
    main()
