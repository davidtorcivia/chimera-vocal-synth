#!/usr/bin/env python3
"""Score the restoration pilot: for each voice and version (conv, eq, all) — PhoneticXeus per-phone misses on the
clips' own spans, the Phase-0 acoustic measures (pie/acoustic_map.py), and UTMOS naturalness (pie/mos.py).

    python pie/pilot_score.py [--pilot DIR] OUT.json
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = Path.home() / "AI/chimera-pie-data"
FULLROOT = DATA / "fullroot"
PY = sys.executable
VERSIONS = ["conv", "tone", "bursts", "s", "eq", "all"]
KEYS = ["kʷ", "k", "p", "t", "s", "r", "j", "w", "e", "o", "u", "i", "a", "m", "n", "h", "χ"]


def gop(name, wavdir, stems):
    lst = Path(wavdir) / "stems.txt"
    lst.write_text("\n".join(stems) + "\n")
    env = dict(os.environ, CHIMERA_PIE_DATA=str(FULLROOT))
    subprocess.run([PY, "-I", str(HERE / "gop.py"), name, "--wavdir", str(wavdir), "--set", "all", "--only", str(lst),
                    "--threads", "10"], capture_output=True, text=True, env=env)
    g = json.load(open(FULLROOT / "runs" / name / "gop.json"))
    pp = g["per_phone"]
    return {"miss": round(100 * (1 - g["broad"]), 1),
            "per": {k: [pp[k]["n"] - pp[k]["broad"], pp[k]["n"]] for k in KEYS if k in pp}}


def utmos(wavdir, name):
    env = dict(os.environ, CHIMERA_PIE_DATA=str(FULLROOT))
    r = subprocess.run([PY, "-I", str(HERE / "mos.py"), "--wavdir", str(wavdir), "--offset", "0", "--jsondir", "rp-utts", name],
                       capture_output=True, text=True, env=env)
    line = [l for l in r.stdout.splitlines() if "MOS" in l]
    return float(line[-1].split("MOS")[1].split()[0]) if line else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--pilot", default=str(DATA / "full/restore-pilot"))
    ap.add_argument("--versions", default="conv,tone,bursts,s,eq,all")
    a = ap.parse_args()
    versions = a.versions.split(",")
    import acoustic_map as AM
    P = Path(a.pilot)
    clips = json.load(open(DATA / "full/restore-pilot-clips.json"))
    import eval as E
    ud = FULLROOT / "runs" / "rp-utts" / "wav"
    ud.mkdir(parents=True, exist_ok=True)
    for stems in clips.values():
        for s_ in stems:
            json.dump(E.utterances(DATA / "full/timing" / f"{s_}.json"), open(ud / f"{s_}.json", "w"), ensure_ascii=False)
    res = {}
    for tag, stems in clips.items():
        male = tag in ("ZAZY", "ANTONIO")
        res[tag] = {}
        for v in versions:
            wd = P / v / tag
            r = gop(f"rp-{v}-{tag}", wd, stems)
            acc = None
            for s in stems:
                m = AM.measure_file(wd / f"{s}.wav", AM.phones(DATA / "full/timing" / f"{s}.json"), male)
                acc = {k: list(vv) for k, vv in m.items()} if acc is None else {k: acc[k] + list(m[k]) for k in acc}
            r["acoustic"] = AM.summarize({"x": acc})["x"]
            r["utmos"] = utmos(wd, f"rp-{v}-{tag}")
            res[tag][v] = r
            print(tag, v, r["miss"], "UTMOS", r["utmos"], flush=True)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
