#!/usr/bin/env python3
"""Render the whole PIE speech corpus with Chimera, as drop-in sources for the speech-to-speech stage.

Each clip × voice of the VTL build (tts-corpus-v1/work) is rendered on its VTL phone timing (renders/*.json),
with that clip's own prosodic variation from the VTL render manifest: the baseline pitch shift (pitch_st, ±2 st
around the voice's baseline) and the accent height (accent_range_st), so the two corpora vary alike.
Output: OUT/renders/CLIP__SPEAKER.wav, 44.1 kHz 16-bit mono, exactly the VTL clip's length (peak 0.5), and
OUT/manifest.jsonl (f0, accent, engine commit). Skips clips already rendered, so a rerun resumes.

    python pie/fullrender.py FULL_DIR OUT [--jobs 12] [--limit N] [--only STEMS]
FULL_DIR holds timing/ (the VTL renders' .json) and vtl_render_manifest.jsonl.
"""
import argparse, json, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval as E

BODY = {"JD3-PIE": "male", "W02-PIE": "female"}
BASE_F0 = {"male": 110.0, "female": 200.0}          # Chimera's PIE voices (Hz)
# the VTL voices' baselines in semitones (pie_vtl speaker_pitch: rest F0 − 3 st); pitch_st varies ±2 around them
BASE_ST = {"JD3-PIE": 79.88, "W02-PIE": 90.38}
LEAD = 0.3                                          # the renderer's silent lead (s)


def commit():
    r = subprocess.run(["git", "-C", str(HERE.parent), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("full")
    ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--extra", default="", help="extra key=value options for the renderer (e.g. lab_glide=0.03)")
    a = ap.parse_args()
    full, out = Path(a.full), Path(a.out)
    (out / "renders").mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in open(full / "vtl_render_manifest.jsonl", encoding="utf-8") if l.strip()]
    if a.only:
        keep = set(open(a.only).read().split())
        recs = [r for r in recs if f"{r['clip']}__{r['speaker']}" in keep]
    if a.limit:
        recs = recs[:a.limit]
    todo = [r for r in recs if not (out / "renders" / f"{r['clip']}__{r['speaker']}.wav").exists()]
    rev = commit()
    print(f"{len(recs)} clips, {len(todo)} to render (engine {rev})", flush=True)

    def one(r):
        stem = f"{r['clip']}__{r['speaker']}"
        body = BODY[r["speaker"]]
        f0 = BASE_F0[body] * 2 ** ((r["pitch_st"] - BASE_ST[r["speaker"]]) / 12)
        timing = full / "timing" / f"{stem}.json"
        dur = json.load(open(timing, encoding="utf-8"))["duration"]
        with tempfile.TemporaryDirectory() as td:
            uj, wav = Path(td) / "u.json", Path(td) / "r.wav"
            json.dump(E.utterances(timing), open(uj, "w"), ensure_ascii=False)
            p = subprocess.run([str(E.BIN), str(uj), str(wav), body, f"f0={f0:.3f}", f"accent_st={r['accent_range_st']}"] + a.extra.split(),
                               capture_output=True, text=True)
            if p.returncode:
                return {"clip": r["clip"], "speaker": r["speaker"], "error": p.stderr[-500:]}
            x, sr = sf.read(wav, dtype="float32")
        x = x[int(LEAD * sr):]
        y = resample_poly(x, 147, 160) if sr == 48000 else x
        n = int(round(dur * 44100))
        y = np.pad(y, (0, max(0, n - len(y))))[:n]
        peak = float(np.max(np.abs(y)))
        if peak < 1e-4:
            return {"clip": r["clip"], "speaker": r["speaker"], "error": "silent"}
        sf.write(out / "renders" / f"{stem}.wav", y / peak * 0.5, 44100, subtype="PCM_16")
        return {"clip": r["clip"], "speaker": r["speaker"], "wav": f"renders/{stem}.wav", "duration_s": round(n / 44100, 4),
                "f0_hz": round(f0, 2), "pitch_st_vtl": r["pitch_st"], "accent_st": r["accent_range_st"], "engine": rev}

    done = 0
    with ThreadPoolExecutor(a.jobs) as ex, open(out / "manifest.jsonl", "a", encoding="utf-8") as man:
        for res in ex.map(one, todo):
            man.write(json.dumps(res, ensure_ascii=False) + "\n")
            done += 1
            if "error" in res:
                print("ERROR", res, flush=True)
            if done % 200 == 0:
                print(f"{done}/{len(todo)}", flush=True)
    print(f"done: {done}", flush=True)


if __name__ == "__main__":
    main()
