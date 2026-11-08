#!/usr/bin/env python3
"""Per-phone scoring with PhoneticXeus (the best universal phone recognizer on PRiSM 2026), on known spans.

    python pie/gop.py NAME [--set devall|test|...] [--wavdir DIR --offset 0] [--run RUN] [--only stems.txt]

Instead of a greedy transcription aligned to the target, every target phone is judged inside its own time span
(we know the timings): the recognizer's frame posteriors (int8 ONNX export allan-valin/phoneticxeus-onnx, CC BY-NC-SA
4.0, used only to measure) are read there.
  hit   the recognizer emits an acceptable token for the phone inside its span (±30 ms): at some frame where most
        of the mass is non-blank, an acceptable token is the most likely non-blank one
  gop   max over the span of log P(acceptable) - log P(non-blank): how sure it is, 0 = certain
  as    what it emits instead, on misses
Acceptable tokens: the phone itself, plus equivalent spellings (dʱ: dʰ d̤; h: h ɦ; w: w u̯; long vowels: short or
long). The vocabulary has no ʁʷ or ɡʷʰ: ʁʷ accepts ʁ, ɡʷʰ accepts ɡʷ ɡʰ (each scored as half the feature set).
Wavs: by default runs/RUN/wav/STEM.wav rendered by eval.py (offset 0.3); --wavdir for other audio (offset 0).
"""
import argparse, glob, json, os, sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval as E

SNAP = glob.glob(str(E.DATA / "hf/models--allan-valin--phoneticxeus-onnx/snapshots/*"))[0]
VOCAB = json.load(open(f"{SNAP}/ipa_vocab.json"))
ID2TOK = {i: t for t, i in VOCAB.items()}
BLANK = VOCAB["<blank>"]
SPECIAL = {i for t, i in VOCAB.items() if t.startswith("<")}

ACCEPT = {
    "g": ["ɡ"], "gʷ": ["ɡʷ"], "gʷʱ": ["ɡʷ", "ɡʰ", "ɡ̤"], "gʱ": ["ɡʰ", "ɡ̤"], "bʱ": ["bʰ", "b̤"], "dʱ": ["dʰ", "d̤"],
    "ɟʱ": ["ɟʰ"], "h": ["h", "ɦ"], "w": ["w", "u̯", "ʍ"], "j": ["j", "i̯"], "ʁ": ["ʁ"], "ʁʷ": ["ʁ"], "r": ["r", "rː"],
    "r̩": ["r̩"], "l̩": ["l̩"], "m̩": ["m̩"], "n̩": ["n̩"],
}


# broad: phonetic neighbours that do not change the PIE phoneme (vowel quality within a height step, length,
# syllabicity, velar for uvular frication); laryngeal and place contrasts of stops stay strict
BROAD = {"e": ["ɛ", "e̞", "eː", "ɛː"], "o": ["ɔ", "o̞", "oː", "ɔː"], "i": ["ɪ", "iː"], "u": ["ʊ", "uː"], "a": ["ɑ", "ɐ", "aː", "ɑː"],
         "ə": ["ɐ", "ɨ", "ɘ", "ɜ"], "χ": ["x", "χː"], "ʁ": ["ɣ", "ʁ"], "ʁʷ": ["ɣ", "ʁ", "w"], "r̩": ["r", "ɾ"], "l̩": ["l", "ɫ"],
         "m̩": ["m"], "n̩": ["n"], "h": ["h", "ɦ"], "w": ["ʋ"], "j": ["ʝ"],
         "c": ["kʲ", "cʲ", "cː", "cʰ"], "ɟ": ["ɡʲ", "ɟː"]}
LABIAL_FOLLOW = {"w", "u̯", "ʷ", "u", "ʊ", "o"}


def accept(seg):
    """Acceptable vocabulary ids for an eval.py segment."""
    ph = seg["pie"].replace("ˈ", "").replace("́", "")
    b = E.base(ph)
    b = {"ɡ": "g", "ɣ": "ʁ"}.get(b, b)
    key = b + ("ʷ" if seg["labial"] else "") + ("ʱ" if seg["breathy"] else "") + ("̩" if seg["syllabic"] else "")
    if key in ACCEPT:
        toks = ACCEPT[key]
    elif b in "aeiouə" and len(b) == 1:
        toks = [b, b + "ː"]
    else:
        toks = [key]
    ids = [VOCAB[t] for t in toks if t in VOCAB]
    if not ids:
        raise SystemExit(f"no token for {key!r} ({seg['pie']})")
    bkey = b + ("\u0329" if seg["syllabic"] else "")
    broad = ids + [VOCAB[t] for t in BROAD.get(key, BROAD.get(bkey, [])) if t in VOCAB]
    # a labialized stop heard as the plain stop followed by [w]: the plain stop's ids, checked with a follower
    plain = []
    if seg["labial"] and b not in ("ʁ",):
        pb = {"g": "ɡ"}.get(b, b)
        plain = [VOCAB[t] for t in ([pb] + ([pb + "ʰ", pb + "̤"] if seg["breathy"] else [])) if t in VOCAB]
    return key, ids, broad, plain


def logits(sess, x16):
    out = sess.run(None, {"wav": x16[None, :].astype(np.float32)})[0][0]
    m = out.max(1, keepdims=True)
    lp = out - m - np.log(np.exp(out - m).sum(1, keepdims=True))
    return lp


def score_utt(lp, n_samples, segs, t_seg0):
    """lp: frames × vocab log posteriors of a cut that starts at t_seg0 (s, utterance time)."""
    hop = n_samples / lp.shape[0] / 16000.0
    p = np.exp(lp)
    nb = 1.0 - p[:, BLANK]
    rows = []
    t = 0.0
    for s in segs:
        key, ids, broad_ids, plain_ids = accept(s)
        a = int(max(0, (s["t0"] - t_seg0 - 0.03) / hop))
        b = int(min(len(p), (s["t1"] - t_seg0 + 0.03) / hop) + 1)
        span = p[a:b]
        if len(span) == 0:
            continue
        pa = span[:, ids].sum(1)
        pnb = np.maximum(nb[a:b], 1e-6)
        gop = float(np.max(np.log(np.maximum(pa, 1e-9)) - np.log(pnb)))
        nbt = span.copy()
        nbt[:, list(SPECIAL)] = 0
        top = nbt.argmax(1)
        live = nb[a:b] > 0.5
        hit = bool(np.any(live & np.isin(top, ids)))
        bhit = hit or bool(np.any(live & np.isin(top, broad_ids)))
        if not bhit and plain_ids:
            # stop then a rounded follower inside the span (+40 ms): kʷ heard as k w
            fol = [VOCAB[t] for t in LABIAL_FOLLOW if t in VOCAB]
            b2 = int(min(len(p), (s["t1"] - t_seg0 + 0.07) / hop) + 1)
            nbt2 = p[a:b2].copy()
            nbt2[:, list(SPECIAL)] = 0
            top2 = nbt2.argmax(1)
            live2 = nb[a:b2] > 0.5
            ks = np.where(live2 & np.isin(top2, plain_ids))[0]
            if len(ks) and np.any(live2[ks[0] + 1:] & np.isin(top2[ks[0] + 1:], fol)):
                bhit = True
        heard = None
        if not hit:
            k = int(np.argmax(nb[a:b]))
            heard = ID2TOK[int(top[k])] if nb[a + k] > 0.5 else "∅"
        rows.append({"key": key, "hit": hit, "broad": bhit, "gop": gop, "as": heard})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--set", default="devall")
    ap.add_argument("--run", default="", help="eval.py run whose wav/ holds the renders (default: NAME)")
    ap.add_argument("--wavdir", default="")
    ap.add_argument("--offset", type=float, default=None)
    ap.add_argument("--only", default="")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    # the int8 export with its ConvInteger layers rewritten as float Conv (the CPU runtime has no int8 ConvInteger);
    # pie/pxeus_patch.py makes it
    sess = ort.InferenceSession(str(E.DATA / "hf/pxeus-convfloat.onnx"), so, providers=["CPUExecutionProvider"])
    files = E.sets(a.set)
    if a.only:
        keep = set(open(a.only).read().split())
        files = [f for f in files if Path(f).stem in keep]
    run = a.run or a.name
    off = a.offset if a.offset is not None else (0.0 if a.wavdir else 0.3)
    allrows, groups = [], {}
    for f in files:
        stem = Path(f).stem
        wav = Path(a.wavdir) / f"{stem}.wav" if a.wavdir else E.DATA / "runs" / run / "wav" / f"{stem}.wav"
        if not wav.exists():
            continue
        x, sr = sf.read(wav, dtype="float32", always_2d=True)
        x = x.mean(1)
        from math import gcd
        g = gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
        u = E.utterances(f)
        grp = "passage" if not stem.startswith("t") else ("corpus-" + ("female" if "W02" in stem else "male"))
        for ut in u["utterances"]:
            # absolute times of each segment
            t, segs = ut["start"], []
            for s in ut["segs"]:
                segs.append(dict(s, t0=t, t1=t + s["dur"]))
                t += s["dur"]
            c0 = ut["start"] - 0.05
            seg = x[max(0, int((c0 + off) * 16000)):int((ut["end"] + off + 0.12) * 16000)]
            if len(seg) < 3200:
                continue
            rows = score_utt(logits(sess, seg), len(seg), segs, c0 if c0 + off >= 0 else -off)
            for r in rows:
                r["file"] = stem
            allrows += rows
            groups.setdefault(grp, []).extend(rows)
    hits = np.mean([r["hit"] for r in allrows])
    bh = np.mean([r["broad"] for r in allrows])
    per = {}
    for r in allrows:
        d = per.setdefault(r["key"], {"n": 0, "hit": 0, "broad": 0, "gop": 0.0, "as": {}})
        d["n"] += 1
        d["hit"] += r["hit"]
        d["broad"] += r["broad"]
        d["gop"] += r["gop"]
        if not r["hit"]:
            d["as"][r["as"]] = d["as"].get(r["as"], 0) + 1
    out = E.DATA / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"name": a.name, "set": a.set, "hit": hits, "broad": bh, "by_group": {g: float(np.mean([r["hit"] for r in v])) for g, v in groups.items()},
               "per_phone": per, "rows": allrows}, open(out / "gop.json", "w"), ensure_ascii=False, indent=0)
    print(f"{a.name} [{a.set}] PXEUS miss strict {100-hits*100:.1f}%  broad {100-bh*100:.1f}%  (" +
          "  ".join(f"{g} {100-np.mean([r['broad'] for r in v])*100:.1f}" for g, v in sorted(groups.items())) + " broad)")
    worst = sorted(per.items(), key=lambda kv: -(kv[1]["n"] - kv[1]["broad"]))[:18]
    print("most missed (broad): " + ", ".join(f"{k} {v['n']-v['broad']}/{v['n']}(" + ",".join(f"{t}{c}" for t, c in sorted(v['as'].items(), key=lambda x: -x[1])[:3]) + ")" for k, v in worst))


if __name__ == "__main__":
    main()
