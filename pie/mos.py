#!/usr/bin/env python3
"""Naturalness proxy: UTMOS22-strong (ONNX export, TigreGotico/utmos-onnx) on each breath group of a run.

    python pie/mos.py RUN [RUN ...]          # runs/RUN/wav/*.wav, cut as eval.py cuts them
    python pie/mos.py --wavdir DIR --offset 0 NAME

Needs onnxruntime, soundfile, scipy (the PIE kokoro venv has them). Prints mean MOS per run and per group.
"""
import argparse, glob, json, os
from math import gcd
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly

DATA = Path(os.environ.get("CHIMERA_PIE_DATA", Path.home() / "AI/chimera-pie-data"))
MODEL = glob.glob(str(DATA / "hf/models--TigreGotico--utmos-onnx/snapshots/*/utmos22_strong.onnx"))[0]


def load16(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    g = gcd(sr, 16000)
    return resample_poly(x, 16000 // g, sr // g).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--wavdir", default="")
    ap.add_argument("--offset", type=float, default=0.3)
    ap.add_argument("--set", default="")
    a = ap.parse_args()
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    for run in a.runs:
        jdir = DATA / "runs" / (run if not a.wavdir else "b0") / "wav"
        wdir = Path(a.wavdir) if a.wavdir else DATA / "runs" / run / "wav"
        groups = {}
        for uj in sorted(jdir.glob("*.json")):
            stem = uj.stem
            wav = wdir / f"{stem}.wav"
            if not wav.exists():
                continue
            x = load16(wav)
            grp = "passage" if not stem.startswith("t") else ("corpus-" + ("female" if "W02" in stem else "male"))
            for ut in json.load(open(uj))["utterances"]:
                seg = x[max(0, int((ut["start"] + a.offset - 0.05) * 16000)):int((ut["end"] + a.offset + 0.12) * 16000)]
                if len(seg) < 8000:
                    continue
                m = float(sess.run(None, {"wave": seg[None, :]})[0][0])
                groups.setdefault(grp, []).append(m)
        allm = [m for v in groups.values() for m in v]
        print(f"{run}: MOS {np.mean(allm):.3f} (n={len(allm)})  " + "  ".join(f"{g} {np.mean(v):.3f}" for g, v in sorted(groups.items())))


if __name__ == "__main__":
    main()
