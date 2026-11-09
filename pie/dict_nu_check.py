#!/usr/bin/env python3
"""Recognizer QA of the site's dictionary audio (the carrier phrase "nu. WORD. nu." it is cut from): PhoneticXeus
transcribes every clip, and a clip is flagged when its transcription starts or ends with "n u" (or "n" + a vowel)
while its own word does not (a carrier "nu" left in), or when it hears under half the word's sounds (truncated).
    python pie/dict_nu_check.py SITE_DIR LABELS.json [--ids a,b,...] [--workers 8] [--out flags.json]
"""
import argparse, glob, json, subprocess, sys, zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
VOW = set("aeiouəɐʌɛɔ")


def shard_job(job):
    import gop as G, wordscore as W
    import onnxruntime as ort
    site, sh, offs, labels = job
    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(G.E.DATA / "hf/pxeus-convfloat.onnx"), so, providers=["CPUExecutionProvider"])
    out = []
    for v in ("m", "f"):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f"{site}/audio/dict/{sh}-{v}.mp3", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                             capture_output=True, check=True).stdout
        y = np.frombuffer(raw, "<f4")
        for i, o in offs.items():
            if v not in o or i not in labels:
                continue
            t0, d = o[v]
            x = y[int(t0 * 16000): int((t0 + d) * 16000)].copy()
            if len(x) < 1600:
                continue
            heard = W.heard_tokens(G.logits(sess, x).argmax(1))
            lab = labels[i].rstrip(".")
            nu_start = len(heard) >= 2 and heard[0] == "n" and heard[1][:1] in VOW and not lab.startswith("n")
            nu_end = len(heard) >= 2 and heard[-2] == "n" and heard[-1][:1] in VOW and not lab.rstrip("̩").endswith(("nu", "n"))
            # truncated: the recognizer hears under half the word's sounds (a syllable cut off)
            nph = sum(1 for c in lab if c.isalpha() and c not in "ʷʱʲ")
            short = nph >= 3 and len(heard) < 0.5 * nph
            if nu_start or nu_end or short:
                out.append({"id": i, "voice": v, "label": labels[i], "heard": " ".join(heard),
                            "where": "start" if nu_start else "end" if nu_end else "truncated"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site"); ap.add_argument("labels"); ap.add_argument("--ids", default="")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--out", default="dict-nu-flags.json")
    a = ap.parse_args()
    labels = json.load(open(a.labels, encoding="utf-8"))
    if a.ids:
        keep = set(a.ids.split(","))
        labels = {k: v for k, v in labels.items() if k in keep}
    jobs = []
    for js in sorted(glob.glob(f"{a.site}/data/dict/audio/[0-9][0-9].json")):
        offs = {k: v for k, v in json.load(open(js)).items() if k in labels}
        if offs:
            jobs.append((a.site, Path(js).stem, offs, labels))
    flags = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(shard_job, jobs):
            flags += r
    by = {}
    for f in flags:
        by[f"{f['voice']} {f['where']}"] = by.get(f"{f['voice']} {f['where']}", 0) + 1
    print(sum(len(j[2]) for j in jobs), "words;", len(flags), "clips with a carrier nu:", by)
    json.dump(flags, open(a.out, "w"), ensure_ascii=False, indent=0)


if __name__ == "__main__":
    main()
