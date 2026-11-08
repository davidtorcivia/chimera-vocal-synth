#!/usr/bin/env python3
"""Voice-quality numbers for WAVs: jitter, shimmer, harmonics-to-noise, and the high band's level against the
low band (4-8 kHz and 8 kHz+ vs 0-4 kHz). python pie/voicequal.py A.wav [B.wav ...]"""
import sys
import numpy as np, parselmouth as pm
from parselmouth.praat import call


def measure(f, skip=0.0):
    s = pm.Sound(f)
    if skip:
        s = s.extract_part(from_time=skip)
    pp = call(s, "To PointProcess (periodic, cc)", 60, 500)
    jit = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
    shim = call([s, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
    hnr = call(call(s, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0), "Get mean", 0, 0)
    sp = s.to_spectrum(); fr = sp.xs(); p = np.abs(sp.values[0] + 1j * sp.values[1]) ** 2
    e = lambda a, b: p[(fr >= a) & (fr < b)].sum()
    return jit, shim, hnr, 10 * np.log10(e(4000, 8000) / e(0, 4000)), 10 * np.log10(e(8000, 20000) / e(0, 4000))


if __name__ == "__main__":
    for f in sys.argv[1:]:
        j, s, h, a, b = measure(f)
        print(f"{f.split('/')[-1][:40]:40} jitter {j:4.2f}%  shimmer {s:4.1f}%  HNR {h:5.1f} dB  4-8k {a:6.1f} dB  8k+ {b:6.1f} dB")
