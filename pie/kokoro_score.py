#!/usr/bin/env python3
"""Recognizer scoring of Kokoro renders of the corpus v2 validation sentences (tts/kokoro/v2/synth_eval.py in the
PIE repo): each word is transcribed by PhoneticXeus inside its own predicted span (±40 ms) and compared with the
word's target phones by edit distance, with the broad equivalences of pie/wordscore.py. The targets are the corpus
timing files (full/timing/<clip>__JD3-PIE.json). The real validation audio (final-v2/wavs, on its corpus timing) is
scored the same way, as the reference.

    python pie/kokoro_score.py EVAL_DIR [RUN ...]       (EVAL_DIR/<run>/spans.jsonl + wavs)
Prints the phone error per run, per voice and male/female; writes EVAL_DIR/scores.json.
"""
import json, sys
from math import gcd
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gop as G
import wordscore as W

FULL = Path.home() / "AI/chimera-pie-data/full"
VOICE = {0: "zazy", 1: "valeria", 2: "antonio", 3: "w02"}
MALE = {0, 2}


def load16(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    g = gcd(sr, 16000)
    return resample_poly(x, 16000 // g, sr // g).astype(np.float32)


def targets(clip):
    for spk in ("JD3-PIE", "W02-PIE"):
        p = FULL / "timing" / f"{clip}__{spk}.json"
        if p.exists():
            return json.load(open(p, encoding="utf-8"))["words"]
    return None


def score(sess, x, spans, words):
    E = N = 0
    for w, (a, b) in zip(words, spans):
        seg = x[max(0, int((a - 0.04) * 16000)):int((b + 0.04) * 16000)]
        tgt = W.target_tokens(w)
        if len(seg) < 1600:
            E += len(tgt); N += len(tgt)          # an unheard word counts as all wrong
            continue
        E += W.edit(W.heard_tokens(G.logits(sess, seg).argmax(1)), tgt)
        N += len(tgt)
    return E, N


def main():
    root = Path(sys.argv[1])
    runs = sys.argv[2:] or sorted(p.name for p in root.iterdir() if (p / "spans.jsonl").exists())
    sess = ort.InferenceSession(str(G.E.DATA / "hf/pxeus-convfloat.onnx"), providers=["CPUExecutionProvider"])
    res, skipped = {}, {}
    rows0 = [json.loads(l) for l in open(root / runs[0] / "spans.jsonl", encoding="utf-8")]
    for run in ["corpus"] + runs:
        acc = {}
        rows = rows0 if run == "corpus" else [json.loads(l) for l in open(root / run / "spans.jsonl", encoding="utf-8")]
        for r in rows:
            clip = r["file"].rsplit("__", 1)[0]
            words = targets(clip)
            if words is None:
                skipped[clip] = "no timing"; continue
            if run == "corpus":
                spans = [(w["start"], w["end"]) for w in words]
                x = load16(FULL / "final-v2/wavs" / r["file"])
            else:
                if len(r["words"]) != len(words):
                    skipped[clip] = f"{len(r['words'])} words vs {len(words)}"; continue
                spans = [(a, b) for _, a, b in r["words"]]
                x = load16(root / run / r["file"])
            e, n = score(sess, x, spans, words)
            for k in (VOICE[r["speaker"]], "male" if r["speaker"] in MALE else "female", "all"):
                acc.setdefault(k, [0, 0]); acc[k][0] += e; acc[k][1] += n
        res[run] = {k: {"error_pct": round(100 * e / n, 2), "edits": e, "phones": n} for k, (e, n) in acc.items()}
        print(f"{run:10s} " + "  ".join(f"{k} {v['error_pct']:5.1f}%" for k, v in res[run].items()), flush=True)
    print("skipped clips:", len(skipped), list(skipped.items())[:5])
    json.dump({"scores": res, "skipped": skipped}, open(root / "scores.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
