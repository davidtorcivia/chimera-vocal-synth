#!/usr/bin/env python3
"""Pitch-accent check: how far accented nuclei rise above the line, per breath group.

    python pie/accent.py NAME:WAVDIR:OFFSET [...] [--set dev] [--limit 40]
For each breath group: F0 (Praat, via parselmouth) at the middle of every nucleus, in semitones; a straight line
fitted through the unaccented nuclei (the declination); accent lift = mean distance of the accented nuclei above
that line. pie_vtl's gate was a lift of at least 1.5 st; its hymn renders measured about +3.1 st at a 4 st target.
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
    ap.add_argument("--limit", type=int, default=40)
    a = ap.parse_args()
    files = E.sets(a.set)
    if a.limit:
        files = [f for f in files if "JD3" in f][: a.limit // 2] + [f for f in files if "W02" in f][: a.limit // 2]
    for spec in a.specs:
        name, wdir, off = spec.split(":")
        off = float(off)
        lifts, n_acc = [], 0
        for f in files:
            stem = Path(f).stem
            wav = Path(wdir) / f"{stem}.wav"
            if not wav.exists():
                continue
            snd = parselmouth.Sound(str(wav))
            pitch = snd.to_pitch(time_step=0.005, pitch_floor=70, pitch_ceiling=450)
            for ut in E.utterances(f)["utterances"]:
                t, pts = ut["start"], []
                for s in ut["segs"]:
                    if E.base(s["pie"]) in "aeiouə" or s["syllabic"]:
                        hz = pitch.get_value_at_time(t + off + s["dur"] / 2)
                        if hz == hz and hz > 0:
                            pts.append((t, 12 * np.log2(hz / 100), s["accent"]))
                    t += s["dur"]
                un = [(x, y) for x, y, acc in pts if not acc]
                ac = [(x, y) for x, y, acc in pts if acc]
                if len(un) < 2 or not ac:
                    continue
                k, b = np.polyfit([x for x, _ in un], [y for _, y in un], 1)
                lifts.append(np.mean([y - (k * x + b) for x, y in ac]))
                n_acc += len(ac)
        L = np.array(lifts)
        print(f"{name:14} groups {len(L):3}  accent lift median {np.median(L):+5.2f} st  mean {L.mean():+5.2f}  under 1.5 st: {np.mean(L < 1.5)*100:4.1f}%")


if __name__ == "__main__":
    main()
