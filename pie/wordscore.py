#!/usr/bin/env python3
"""Word-by-word phone scoring that works for any timing (e.g. the site's Kokoro voice): each word's span is
transcribed by PhoneticXeus (greedy CTC on the word ±40 ms) and compared with the word's target phones by edit
distance, with broad equivalences (vowel quality neighbours, length, syllabicity, x for χ, kʲ for c, k w for kʷ,
dʱ = dʰ = d̤). Reports the error rate (edits / target phones) per version.

    python pie/wordscore.py TARGET.json NAME:WAV:TIMING [...]
TARGET.json: a pie_vtl render JSON (words with phones). TIMING: "same" (the target JSON's word times, +offset
OFFSET with NAME:WAV:same@0.3) or a site.json hymn timing key such as "site:m".
"""
import json, sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gop as G
import onnxruntime as ort

FOLD = {"ɛ": "e", "e̞": "e", "ɔ": "o", "o̞": "o", "ɪ": "i", "ʊ": "u", "ɑ": "a", "ɐ": "a", "ɜ": "ə", "ɨ": "ə", "ɘ": "ə",
        "x": "χ", "ɣ": "ʁ", "ɦ": "h", "kʲ": "c", "cʲ": "c", "ɡʲ": "ɟ", "ɾ": "r", "u̯": "w", "i̯": "j", "ʍ": "w"}


def canon(tok):
    t = tok.replace("ː", "").replace("̩", "").replace("̍", "")
    t = t.replace("̤", "ʰ").replace("ʱ", "ʰ")          # b̤ → bʰ (murmur), ʱ → ʰ
    if t and t[0] in "bdɡɟ" and t.endswith("ʰ"):
        t = t[:-1] + "ʱ"                                    # voiced + breath = murmured
    return FOLD.get(t, t)


def target_tokens(word):
    out = []
    for p in word["phones"]:
        b = G.E.base(p["phone"])
        b = {"g": "ɡ", "ɣ": "ʁ"}.get(b, b)
        t = b + ("ʷ" if p["labial"] else "") + ("ʱ" if p["breathy"] else "")
        out.append(canon(t))
    return out


def heard_tokens(ids):
    toks, prev = [], -1
    for i in ids:
        if i != prev and i not in G.SPECIAL:
            toks.append(canon(G.ID2TOK[int(i)]))
        prev = i
    # a plain velar followed by w is a labiovelar
    out = []
    for t in toks:
        if out and t == "w" and out[-1] in ("k", "ɡ", "ɡʱ", "ʁ"):
            out[-1] = out[-1][:1] + "ʷ" + out[-1][1:]
            continue
        out.append(t)
    return out


def edit(a, b):
    D = np.zeros((len(a) + 1, len(b) + 1), int)
    D[:, 0] = range(len(a) + 1)
    D[0, :] = range(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    return int(D[-1, -1])


def main():
    target = json.load(open(sys.argv[1]))["words"]
    sess = ort.InferenceSession(str(G.E.DATA / "hf/pxeus-convfloat.onnx"), providers=["CPUExecutionProvider"])
    site = json.load(open(Path.home() / "AI/PIE-site/data/site.json"))["hymn"]["timing"]
    for spec in sys.argv[2:]:
        name, wav, timing = spec.split(":", 2)
        off = 0.0
        if timing.startswith("same"):
            off = float(timing.split("@")[1]) if "@" in timing else 0.0
            spans = [(w["start"] + off, w["end"] + off) for w in target]
        else:
            spans = [(w["start"], w["end"]) for w in site[timing.split(":")[1]]]
        x, sr = sf.read(wav, dtype="float32", always_2d=True)
        x = x.mean(1)
        from math import gcd
        g = gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
        E = N = 0
        for w, (a, b) in zip(target, spans):
            seg = x[max(0, int((a - 0.04) * 16000)):int((b + 0.04) * 16000)]
            if len(seg) < 1600:
                continue
            heard = heard_tokens(G.logits(sess, seg).argmax(1))
            tgt = target_tokens(w)
            E += edit(heard, tgt)
            N += len(tgt)
        print(f"{name:22} word-span phone error {E / N * 100:5.1f}%  ({E}/{N})")


if __name__ == "__main__":
    main()
