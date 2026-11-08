#!/usr/bin/env python3
"""Vowel loudness: each vowel's level (middle half, dB) against the median vowel of its clip, by vowel, for one or
more eval.py runs. The corpus QC calls a nucleus weak at 8 dB under the median.

    python pie/vowelloud.py RUN [RUN ...] [--set dev]
"""
import argparse, collections, json, sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval as E

VOW = "aeiouə"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--set", default="dev")
    a = ap.parse_args()
    for run in a.runs:
        wd = E.DATA / "runs" / run / "wav"
        rel = collections.defaultdict(list)
        for w in sorted(wd.glob("*.wav")):
            d = json.load(open(E.DATA / "corpus" / f"{w.stem}.json"))
            x, sr = sf.read(w, dtype="float32")
            lv = []
            for wd_ in d["words"]:
                for p in wd_["phones"]:
                    b = E.base(p["phone"])
                    if b[:1] in VOW and len(b) == 1:
                        a0, a1 = p["start"] + 0.3, p["end"] + 0.3
                        m = 0.25 * (a1 - a0)
                        s = x[int((a0 + m) * sr):int((a1 - m) * sr)]
                        if len(s) > 0.01 * sr:
                            lv.append((b, 20 * np.log10(np.sqrt(np.mean(s ** 2)) + 1e-9)))
            if not lv:
                continue
            med = float(np.median([v for _, v in lv]))
            for b, v in lv:
                rel[b].append(v - med)
        print(run, "  ".join(f"{b} {np.median(rel[b]):+.1f} dB ({100 * np.mean(np.array(rel[b]) < -8):.0f}% weak)" for b in VOW if rel[b]))


if __name__ == "__main__":
    main()
