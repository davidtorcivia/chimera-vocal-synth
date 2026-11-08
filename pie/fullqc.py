#!/usr/bin/env python3
"""The corpus pipeline's QC gate (PIE repo tts/vtl/qc.py and corpus_pipeline._qc's floors) on the Chimera renders
of the whole corpus, each against its own VTL phone timing.

    python pie/fullqc.py FULL_DIR RENDER_DIR [--jobs 14]       # writes RENDER_DIR/qc.jsonl, prints the summary
"""
import argparse, collections, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

QC = Path.home() / "AI/PIE/tts/vtl/qc.py"


def one(args):
    full, rd, stem = args
    wav = rd / "renders" / f"{stem}.wav"
    r = subprocess.run([sys.executable, str(QC), str(wav), "--timing", str(full / "timing" / f"{stem}.json"), "--ref", str(wav),
                        "--final-h", "murmur", "--min-lift", "1.5"], capture_output=True, text=True, cwd=QC.parent)
    try:
        q = json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"stem": stem, "pass": False, "why": ["qc error: " + r.stderr[-200:]]}
    why = []
    lost_h = [x for x in q["h_share"] if x < 0.25]
    lost_c = [x for x in q["chi_share"] if x < 0.30]
    if lost_h: why.append("h lost")
    if lost_c: why.append("chi lost")
    if q["voiced_share"] < 0.97: why.append("voiced")
    if not q["accent_lift_st"] >= 1.5: why.append("accent lift")
    if len(q["weak_vowels"]) > 1: why.append("weak vowels")
    return {"stem": stem, "pass": not why, "why": why, "accent_lift_st": q["accent_lift_st"], "voiced_share": q["voiced_share"],
            "weak": q["weak_vowels"], "h_share": q["h_share"], "chi_share": q["chi_share"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("full")
    ap.add_argument("renders")
    ap.add_argument("--jobs", type=int, default=14)
    a = ap.parse_args()
    full, rd = Path(a.full), Path(a.renders)
    stems = sorted(p.stem for p in (rd / "renders").glob("*.wav"))
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(one, [(full, rd, s) for s in stems]))
    with open(rd / "qc.jsonl", "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    why = collections.Counter(w for r in res for w in r["why"])
    lifts = sorted(r["accent_lift_st"] for r in res if r.get("accent_lift_st") is not None)
    print(f"{sum(r['pass'] for r in res)}/{len(res)} pass the corpus gate; failures by reason: {dict(why)}")
    if lifts:
        print(f"accent lift median {lifts[len(lifts) // 2]:.2f} st")


if __name__ == "__main__":
    main()
