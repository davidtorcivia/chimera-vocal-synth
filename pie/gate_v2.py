#!/usr/bin/env python3
"""Corpus v2 QC gate, on the restored clips (full/restored-v2) and the QC measures from pie/fullqc.py (rqc-<TAG>/qc.jsonl).
  - no h/χ brightness floors (built for VTL's weak h; the converter brightens h, and χ comes out better than its source)
  - voicing: a creak-tolerant re-check (Praat cross-correlation pitch, floor 50 Hz male / 90 Hz female) of every nucleus
    of a clip the old meter flagged; the clip fails if 2+ nuclei are under 30 % voiced
  - accent lift >= 1.5 st (unchanged); more than 2 weak vowels fails (was 1)
Writes full/gate-v2.json {"TAG/stem": {"pass": bool, "why": [...]}} and prints the pass rates.
    python pie/gate_v2.py [--workers 12]
"""
import argparse, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

FULL = Path.home() / "AI/chimera-pie-data/full"
NUC = ("V", "Vl", "VL", "Ns", "Ls", "Rs")


def recheck(job):
    import parselmouth
    tag, stem, male = job
    snd = parselmouth.Sound(str(FULL / "restored-v2" / tag / f"{stem}.wav"))
    pit = snd.to_pitch_cc(time_step=0.002, pitch_floor=50 if male else 90, pitch_ceiling=350 if male else 500)
    d = json.load(open(FULL / "timing" / f"{stem}.json"))
    bad = 0
    for w in d["words"]:
        for p in w["phones"]:
            if p["kind"] in NUC and p["end"] - p["start"] > 0.03:
                ts = np.arange(p["start"], p["end"], 0.002)
                v = np.mean([(pit.get_value_at_time(t) or 0) > 0 for t in ts])
                bad += v < 0.30
    return tag, stem, int(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    qc, jobs = {}, []
    for tag in ("ZAZY", "VALERIA", "ANTONIO", "W02"):
        for l in open(FULL / f"rqc-{tag}/qc.jsonl"):
            r = json.loads(l)
            qc[f"{tag}/{r['stem']}"] = r
            if "voiced" in r["why"]:
                jobs.append((tag, r["stem"], tag in ("ZAZY", "ANTONIO")))
    unv = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for tag, stem, bad in ex.map(recheck, jobs, chunksize=8):
            unv[f"{tag}/{stem}"] = bad
    gate, stats = {}, {}
    for key, r in qc.items():
        why = []
        if "accent_lift_st" not in r:
            why.append("qc error")
        else:
            if not r["accent_lift_st"] >= 1.5:
                why.append("accent lift")
            if len(r.get("weak", [])) > 2:
                why.append("weak vowels")
            if unv.get(key, 0) >= 2:
                why.append("unvoiced nuclei")
        gate[key] = {"pass": not why, "why": why, "unvoiced_nuclei": unv.get(key, 0)}
        t = key.split("/")[0]
        s = stats.setdefault(t, {"n": 0, "pass": 0, "why": {}})
        s["n"] += 1; s["pass"] += not why
        for w in why:
            s["why"][w] = s["why"].get(w, 0) + 1
    json.dump(gate, open(FULL / "gate-v2.json", "w"), indent=0)
    for t, s in stats.items():
        print(f"{t}: {s['pass']}/{s['n']} pass ({100 * s['pass'] / s['n']:.1f} %)  {s['why']}")
    tot = sum(s["pass"] for s in stats.values()); n = sum(s["n"] for s in stats.values())
    print(f"all: {tot}/{n} ({100 * tot / n:.1f} %)")


if __name__ == "__main__":
    main()
