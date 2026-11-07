#!/usr/bin/env python3
"""Re-score saved runs from their recognizer output (no re-rendering, no recognizer).

    python pie/rescore.py RUN [RUN ...]
Prints fine (as eval.py) and glide-fair: fine with w = u and j = i (after a vowel [w] and [u̯] are one sound,
and the recognizer writes either), so a rounder /w/ heard as "uw" is not counted twice.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval as E

GF = dict(E.FINE)
GF.update({'w': 'u', 'j': 'i'})


def score(run):
    d = E.DATA / "runs" / run
    res = json.load(open(d / "scores.json"))
    utts = {}
    tot = {"fine": [0, 0], "gf": [0, 0]}
    for it in res["items"]:
        f = it["file"]
        if f not in utts:
            utts[f] = json.load(open(d / "wav" / f"{f}.json"))["utterances"]
        segs = utts[f][it["utt"]]["segs"]
        rec = list(it["rec"])
        # rec was joined phones; re-split into characters is fine for norm() (it works per character)
        for name, table in (("fine", E.FINE), ("gf", GF)):
            tgt, _ = E.target_seq(segs, E.FINE if table is GF else table)
            if table is GF:
                tgt = [GF.get(x, x) for x in tgt]  # FINE's targets already carry kʷ's w
            r = E.norm(rec, table)
            if table is GF:
                r = [GF.get(x, x) for x in r]
            e, _, _ = E.align(r, tgt)
            tot[name][0] += e
            tot[name][1] += len(tgt)
    return {k: v[0] / v[1] for k, v in tot.items()}


for run in sys.argv[1:]:
    s = score(run)
    print(f"{run:10} fine {s['fine']*100:.1f}  glide-fair {s['gf']*100:.1f}")
