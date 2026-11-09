#!/usr/bin/env python3
"""v2's target mask: the stretches of each training pair where the repaired paid conversion (the converter's target) is
measured to be wrong more often than the Chimera render it was converted from, so the converter is not taught to copy
the target's error there. Held-out PhoneticXeus misses, paid vs Chimera (pie/sts_eval.py, 2026-11-09): k 34 vs 8 %,
kʷ 45 vs 20, j 49 vs 29, r 36 vs 22, u 27 vs 17; gʷ, gʷʱ (with kʷ) and r̩ (with r) share the mechanism. Those spans
cover 8 % of the training audio. Inside them v2 drops the copy losses (multi-scale mel and feature matching) and keeps
the adversarial loss, so the sound is still judged against the natural voices while what is said comes from the input.

Phone keys are pie/gop.py's (base + ʷ labial + ʱ breathy + syllabic mark), from the corpus timing files, which hold for
every voice (conversion keeps the timing sample for sample).
    python pie/sts/mask.py OUT.json [--margin 0.010]      ->  {"TAG/clip__SPK": [[start_sample, end_sample], ...]}
"""
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import eval as E                                                  # noqa: E402

FULL = Path.home() / "AI/chimera-pie-data/full"
SPK = {"ZAZY": "JD3-PIE", "ANTONIO": "JD3-PIE", "VALERIA": "W02-PIE", "W02": "W02-PIE"}
MASK = {"k", "kʷ", "gʷ", "gʷʱ", "j", "r", "r̩", "u"}
SR = 44100


def key(p):
    b = E.base(p["phone"]); b = {"ɡ": "g", "ɣ": "ʁ"}.get(b, b)
    return b + ("ʷ" if p["labial"] else "") + ("ʱ" if p["breathy"] else "") + ("̩" if p["kind"] in ("Ns", "Ls", "Rs") else "")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--margin", type=float, default=0.010)
    a = ap.parse_args()
    out, tot, cov = {}, 0.0, 0.0
    for split in ("train", "val"):
        for l in open(FULL / f"final-v2/{split}_list.txt", encoding="utf-8"):
            clip, tag = l.split("|")[0][:-4].rsplit("__", 1)
            s = f"{clip}__{SPK[tag]}"
            d = json.load(open(FULL / "timing" / f"{s}.json"))
            iv = []
            for w in d["words"]:
                for p in w["phones"]:
                    if key(p) in MASK:
                        a0, a1 = int((p["start"] - a.margin) * SR), int((p["end"] + a.margin) * SR)
                        if iv and a0 <= iv[-1][1]:
                            iv[-1][1] = max(iv[-1][1], a1)
                        else:
                            iv.append([max(0, a0), a1])
            out[f"{tag}/{s}"] = iv
            if split == "train":
                tot += d["duration"]; cov += sum(b - a_ for a_, b in iv) / SR
    json.dump(out, open(a.out, "w"), ensure_ascii=False)
    print(len(out), "pairs;", f"{100 * cov / tot:.1f} % of the training audio masked (margin {a.margin * 1000:.0f} ms)")


if __name__ == "__main__":
    main()
