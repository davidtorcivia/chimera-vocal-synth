#!/usr/bin/env python3
"""Restoration v2, the whole corpus: the chain the owner chose in the pilot (pie/RESTORATION.md), post-processing only.

  1. mastering EQ (per-voice curve, boosts only)            all voices
  2. s correction (cut 2.5-5 kHz / lift 6.5-10 kHz in s)    Valeria 7/6 dB, W02 9/8 dB
  3. bursts: half of each burst's shortfall vs the source   all voices
  4. re-trill (PIE repo tts/vtl/retrill.py, as in corpus v1) all voices
  5. breathy release after bʰ dʰ gʰ, noise −10 dB          Valeria, W02

    python pie/restore_full.py [--workers 12] [--limit N]
Reads full/eleven-v2/<TAG>/, the Chimera sources full/chimera-v2/renders/ and the timings full/timing/; writes
full/restored-v2/<TAG>/<stem>.wav (44.1 kHz 16-bit, the conversion's length, its peak capped at 0.95) and
full/restored-v2/manifest.jsonl. Skips files already written.
"""
import argparse, json, os, sys, tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path.home() / "AI/PIE/tts/vtl"))
FULL = Path.home() / "AI/chimera-pie-data/full"
OUT = FULL / "restored-v2"
CURVES = json.load(open(HERE / "restore-lab/eq-curves.json"))
S_FIX = {"VALERIA": (7.0, 6.0), "W02": (9.0, 8.0)}
MURMUR = {"VALERIA": -10.0, "W02": -10.0}


def one(job):
    import restore2 as R
    import retrill as RT
    tag, stem = job
    out = OUT / tag / f"{stem}.wav"
    if out.exists():
        return {"tag": tag, "stem": stem, "skipped": True}
    x, sr = sf.read(FULL / "eleven-v2" / tag / f"{stem}.wav", dtype="float64")
    src, _ = sf.read(FULL / "chimera-v2/renders" / f"{stem}.wav", dtype="float64")
    timing = FULL / "timing" / f"{stem}.json"
    ph = R.phones(timing)
    peak = min(0.95, float(np.max(np.abs(x))))
    c = R.eq_voice(x, sr, np.array(CURVES[tag]["edges"]), np.array(CURVES[tag]["gains_db"]))
    if tag in S_FIX:
        c = R.eq_s(c, sr, ph, *S_FIX[tag])
    c = R.contrast_transfer(c, src, sr, R.burst_spans(ph, alpha=0.5))
    c = c / max(1e-9, np.max(np.abs(c))) * peak
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "a.wav", Path(td) / "b.wav"
        sf.write(a, c, sr, subtype="FLOAT")
        n_tr = RT.retrill(str(a), str(timing), str(b))
        c, _ = sf.read(b, dtype="float64")
    if tag in MURMUR:
        c = R.fix_murmur_noise(c, sr, ph, noise_db=MURMUR[tag])
    c = c / max(1e-9, np.max(np.abs(c))) * peak
    c = np.pad(c, (0, max(0, len(x) - len(c))))[: len(x)]
    sf.write(out, c, sr, subtype="PCM_16")
    return {"tag": tag, "stem": stem, "wav": f"{tag}/{stem}.wav", "seconds": round(len(x) / sr, 4), "trills_recut": n_tr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    jobs = []
    for tag in ("ZAZY", "VALERIA", "ANTONIO", "W02"):
        (OUT / tag).mkdir(parents=True, exist_ok=True)
        jobs += [(tag, p.stem) for p in sorted((FULL / "eleven-v2" / tag).glob("*.wav"))]
    if a.limit:
        jobs = jobs[: a.limit]
    print(len(jobs), "clips", flush=True)
    done = err = 0
    with ProcessPoolExecutor(a.workers) as ex, open(OUT / "manifest.jsonl", "a", encoding="utf-8") as man:
        for r in ex.map(_safe, jobs, chunksize=4):
            if r.get("skipped"):
                continue
            man.write(json.dumps(r, ensure_ascii=False) + "\n")
            done += 1
            err += "error" in r
            if "error" in r:
                print("ERROR", r, flush=True)
            if done % 500 == 0:
                print(f"{done} done, {err} errors", flush=True)
    print(f"done: {done}, errors {err}", flush=True)


def _safe(job):
    try:
        return one(job)
    except Exception as e:
        return {"tag": job[0], "stem": job[1], "error": repr(e)[:300]}


if __name__ == "__main__":
    main()
