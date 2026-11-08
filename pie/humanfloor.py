#!/usr/bin/env python3
"""Human floor for the whole-transcription score: run PhoneticXeus on real recorded speech (VoxAngeles, UCLA
Phonetics Lab Archive, 95 languages, audited IPA) and score it exactly as pie/wordscore.py scores our renders
(greedy CTC, broad folding, edit distance / target phones).

    python pie/humanfloor.py PARQUET [N]
"""
import io, sys, random
from pathlib import Path
import numpy as np, soundfile as sf
from math import gcd
from scipy.signal import resample_poly
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gop as G, wordscore as W
import onnxruntime as ort

VOC = sorted((t for t in G.VOCAB if t and not t.startswith("<") and t not in ("|", "[", "]")), key=len, reverse=True)


def tokenize(s):
    out, i = [], 0
    s = s.replace(" ", "")
    while i < len(s):
        for t in VOC:
            if s.startswith(t, i):
                out.append(W.canon(t)); i += len(t); break
        else:
            if out:                       # unknown diacritic: attach to the previous phone
                out[-1] = W.canon(out[-1] + s[i])
            i += 1
    return out


def main():
    t = pq.read_table(sys.argv[1]).to_pylist()
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    random.Random(0).shuffle(t)
    sess = ort.InferenceSession(str(G.E.DATA / "hf/pxeus-convfloat.onnx"), providers=["CPUExecutionProvider"])
    E = N = 0
    per_lang = {}
    for r in t[:n]:
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
        x = x.mean(1); g = gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
        if len(x) < 1600:
            continue
        heard = W.heard_tokens(G.logits(sess, x).argmax(1))
        tgt = tokenize(r["transcript"])
        if not tgt:
            continue
        e = W.edit(heard, tgt)
        E += e; N += len(tgt)
        L = per_lang.setdefault(r["id"][:3], [0, 0]); L[0] += e; L[1] += len(tgt)
    print(f"VoxAngeles human speech, {n} words, {N} phones: whole-transcription phone error {E / N * 100:.1f}%")
    langs = sorted(per_lang.items(), key=lambda kv: kv[1][0] / kv[1][1])
    print("best:", ", ".join(f"{k} {v[0]/v[1]*100:.0f}%" for k, v in langs[:6] if v[1] > 30))
    print("worst:", ", ".join(f"{k} {v[0]/v[1]*100:.0f}%" for k, v in langs[-6:] if v[1] > 30))


if __name__ == "__main__":
    main()
