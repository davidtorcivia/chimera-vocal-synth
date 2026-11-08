#!/usr/bin/env python3
"""h and χ brightness as the corpus QC measures it (tts/vtl/qc.py band_share: 2-8 kHz energy share, h median over
its span, χ mid ±20 ms) for eval.py runs. python pie/hshare.py RUN [RUN ...]"""
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path.home() / "AI/PIE/tts/vtl"))
import eval as E
import qc as Q

for run in sys.argv[1:]:
    hs, cs = [], []
    for w in sorted((E.DATA / "runs" / run / "wav").glob("*.wav")):
        d = json.load(open(E.DATA / "corpus" / f"{w.stem}.json"))
        x, sr = sf.read(w, dtype="float32")
        for wd in d["words"]:
            for p in wd["phones"]:
                b = E.base(p["phone"])
                a0, a1 = p["start"] + 0.3, p["end"] + 0.3
                if b == "h":
                    hs.append(Q.band_share(x, sr, a0, a1))
                elif b == "χ":
                    m = 0.5 * (a0 + a1)
                    cs.append(Q.band_share(x, sr, m - 0.02, m + 0.02))
    hs, cs = np.array([h for h in hs if h == h]), np.array([c for c in cs if c == c])
    print(f"{run}: h share median {np.median(hs):.2f} ({np.mean(hs < 0.25) * 100:.0f}% under 0.25)  χ median {np.median(cs):.2f} ({np.mean(cs < 0.30) * 100:.0f}% under 0.30)")
