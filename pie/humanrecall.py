#!/usr/bin/env python3
"""Does the recognizer hear a sound in real speech? For each target sound, the VoxAngeles words whose audited
transcription contains it, and the share whose recognized phones contain it too (with the same broad
equivalences as pie/gop.py). python pie/humanrecall.py PARQUET"""
import io, sys, collections
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gop as G, humanfloor as H
import onnxruntime as ort

TARGETS = {"bʱ": ["bʱ", "b̤"], "dʱ": ["dʱ", "d̤"], "ɡʱ": ["ɡʱ", "ɡ̤"], "c": ["c", "kʲ", "cʰ"], "ɟ": ["ɟ", "ɡʲ"],
           "χ": ["χ", "x"], "ʁ": ["ʁ", "ɣ"], "h": ["h", "ɦ"], "ə": ["ə", "ɐ", "ɨ", "ɘ", "ɜ"], "j": ["j"], "w": ["w"],
           "r": ["r"], "kʷ": ["kʷ"], "ɡʷ": ["ɡʷ"], "l": ["l", "ɫ"], "t": ["t", "t̪"], "d": ["d", "d̪"], "s": ["s"],
           "e": ["e", "ɛ", "e̞", "eː"], "i": ["i", "ɪ", "iː"], "u": ["u", "ʊ", "uː"], "o": ["o", "ɔ", "oː"], "n": ["n", "n̪"], "p": ["p"]}


def raw_tokens(s):
    out, i = [], 0
    while i < len(s):
        for t in H.VOC:
            if s.startswith(t, i):
                out.append(t); i += len(t); break
        else:
            i += 1
    return out


def main():
    t = pq.read_table(sys.argv[1]).to_pylist()
    sess = ort.InferenceSession(str(G.E.DATA / "hf/pxeus-convfloat.onnx"), providers=["CPUExecutionProvider"])
    hit, n, langs = collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter)
    for r in t:
        tg = raw_tokens(r["transcript"].replace(" ", ""))
        want = [k for k, alts in TARGETS.items() if any(a in tg for a in alts)]
        want = [k for k in want if n[k] < 150]
        if not want:
            continue
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
        x = x.mean(1); g = gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
        if len(x) < 1600:
            continue
        ids = G.logits(sess, x).argmax(1)
        heard, prev = [], -1
        for i in ids:
            if i != prev and i not in G.SPECIAL:
                heard.append(G.ID2TOK[int(i)])
            prev = i
        hs = "".join(heard)
        for k in want:
            n[k] += 1; langs[k][r["id"][:3]] += 1
            hit[k] += any(a in heard for a in TARGETS[k]) or (k in ("kʷ", "ɡʷ") and k[0] + "w" in hs.replace("ɡ", "ɡ"))
    for k in TARGETS:
        if n[k]:
            print(f"{k:3} heard in {hit[k]:3}/{n[k]:3} real words ({hit[k]/n[k]*100:3.0f}%)  languages: {', '.join(f'{l} {c}' for l, c in langs[k].most_common(4))}")


if __name__ == "__main__":
    main()
