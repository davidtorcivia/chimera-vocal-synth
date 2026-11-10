#!/usr/bin/env python3
"""Stop-release windows for the converter's burst loss (train_e2e.py --bursts): Chimera's stop releases are low-frequency
pops (release centroid p 279 Hz, t 1.4 kHz, c 0.6 kHz) that the paid conversion turns into crisp bursts (1.2, 3.0, 3.1 kHz);
v3 keeps part of the pop, which the owner hears as a low click (2026-11-10). For p t c b d g (k, kʷ and gʷ stay under the
accuracy mask), the window from 5 ms before to 25 ms after each stop's end, from the corpus timing files.
    python pie/sts/bursts.py OUT.json   ->  {"TAG/clip__SPK": [[start_sample, end_sample], ...]}
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import eval as E                                                  # noqa: E402
from mask import key, FULL, SPK, SR                              # noqa: E402
STOPS = {"p", "t", "c", "b", "d", "g", "ɟ", "dʱ", "bʱ", "gʱ", "ɟʱ"}
out, n = {}, 0
for split in ("train", "val"):
    for l in open(FULL / f"final-v2/{split}_list.txt", encoding="utf-8"):
        clip, tag = l.split("|")[0][:-4].rsplit("__", 1); s = f"{clip}__{SPK[tag]}"
        d = json.load(open(FULL / "timing" / f"{s}.json"))
        iv = [[int((p["end"] - 0.005) * SR), int((p["end"] + 0.025) * SR)] for w in d["words"] for p in w["phones"] if p["kind"] == "S" and key(p) in STOPS]
        out[f"{tag}/{s}"] = iv; n += len(iv)
json.dump(out, open(sys.argv[1], "w"))
print(len(out), "pairs,", n, "stop releases")
