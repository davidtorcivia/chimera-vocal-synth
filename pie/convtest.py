#!/usr/bin/env python3
"""Render a list of corpus sentences with given PIE options, send them through the ElevenLabs voice changer with
the corpus settings, and score the converted audio with PhoneticXeus (pie/gop.py).

    python pie/convtest.py NAME STEMS.txt [--set dev] [--cap 1500] [--extra "key=value ..."]

Writes $CHIMERA_PIE_DATA/convtests/NAME/{src,conv}/*.wav and the scores (runs/ct-NAME-raw, ct-NAME-conv).
Needs ~/.config/elevenlabs/api_key and the PIE repo's tts/vtl/corpus_pipeline.py (its request code and settings).
Stops at --cap credits.
"""
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval as E

PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("stems")
    ap.add_argument("--set", default="dev")
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    out = E.DATA / "convtests" / a.name
    (out / "src").mkdir(parents=True, exist_ok=True)
    (out / "conv").mkdir(exist_ok=True)
    run = f"ct-{a.name}"
    subprocess.run([PY, "-I", str(HERE / "eval.py"), run, "--set", a.set, "--only", a.stems, "--jobs", "12",
                    "--extra", a.extra, "--render-only"], check=True, capture_output=True)
    stems = open(a.stems).read().split()
    for s in stems:
        x, sr = sf.read(E.DATA / "runs" / run / "wav" / f"{s}.wav", dtype="float32")
        d = json.load(open(E.DATA / "corpus" / f"{s}.json"))["duration"]
        x = x[int(0.3 * sr):]
        y = resample_poly(x, 147, 160)
        n = int(round(d * 44100)) + int(0.25 * 44100)
        y = np.pad(y, (0, max(0, n - len(y))))[:n]
        sf.write(out / "src" / f"{s}.wav", y / np.max(np.abs(y)) * 0.5, 44100, subtype="PCM_16")
    sys.path.insert(0, str(Path.home() / "AI/PIE/tts/vtl"))
    import corpus_pipeline as cp
    key = (Path.home() / ".config/elevenlabs/api_key").read_text().strip()
    total = 0
    with open(out / "eleven_log.jsonl", "a") as log:
        for s in stems:
            if (out / "conv" / f"{s}.wav").exists():
                continue
            m = {"vtl_wav": f"src/{s}.wav", "eleven_wav": f"conv/{s}.wav", "speaker": s.split("__")[1]}
            r = cp._convert(out, m, key)
            log.write(json.dumps(r) + "\n")
            if "error" in r:
                raise SystemExit(f"conversion error: {r}")
            total += r["cost"]
            if total >= a.cap:
                print(f"credit cap {a.cap} reached")
                break
    print(f"{a.name}: converted, {total} credits")
    for kind in ("raw", "conv"):
        wd = out / ("src" if kind == "raw" else "conv")
        r = subprocess.run([PY, "-I", str(HERE / "gop.py"), f"{run}-{kind}", "--wavdir", str(wd), "--set", a.set,
                            "--only", a.stems, "--threads", "14"], capture_output=True, text=True)
        print([l for l in r.stdout.splitlines() if "PXEUS" in l][0])


if __name__ == "__main__":
    main()
