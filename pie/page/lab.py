#!/usr/bin/env python3
"""Build the lab-notebook data for the listening page: for each conversion test in lab.json, encode the raw and
converted audio of the test sentences as mp3 under PAGE/c/<test>/ and collect the scores from the runs.

    python pie/page/lab.py PAGE_DIR          # writes PAGE_DIR/../lab-data.json
"""
import json, subprocess, sys
from pathlib import Path

DATA = Path.home() / "AI/chimera-pie-data"
HERE = Path(__file__).resolve().parent


def miss(rows):
    return round(100 * sum(not r["broad"] for r in rows) / max(1, len(rows)), 1)


def main():
    page = Path(sys.argv[1])
    lab = json.load(open(HERE / "lab.json"))
    stems = open(DATA / "pilot/chi_stems.txt").read().split()
    sents = []
    for s in stems:
        d = json.load(open(DATA / "corpus" / f"{s}.json"))
        sents.append({"id": s, "voice": "female" if "W02" in s else "male", "dur": round(d["duration"], 1),
                      "words": " ".join(w["word"] for w in d["words"]),
                      "ipa": " ".join("".join(p["phone"] for p in w["phones"]) for w in d["words"]), "px": {}})
    tests = []
    for t in lab["tests"]:
        n = t["name"]
        rr, cr = t.get("raw_run", f"ct-{n}-raw"), t.get("conv_run", f"ct-{n}-conv")
        rd = DATA / t.get("raw_dir", f"convtests/{n}/src")
        cd = DATA / t.get("conv_dir", f"convtests/{n}/conv")
        if not (DATA / "runs" / cr / "gop.json").exists():
            continue
        out = page / "c" / n
        out.mkdir(parents=True, exist_ok=True)
        for s in stems:
            for kind, src in (("raw", rd), ("conv", cd)):
                mp3 = out / f"{s}-{kind}.mp3"
                if not mp3.exists():
                    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src / f"{s}.wav"), "-af",
                                    "loudnorm=I=-20:TP=-2", "-ac", "1", "-ar", "44100", "-b:a", "96k", str(mp3)], check=True)
        R = json.load(open(DATA / "runs" / rr / "gop.json"))["rows"]
        C = json.load(open(DATA / "runs" / cr / "gop.json"))["rows"]
        ph = lambda rows, p: miss([r for r in rows if r["key"] == p])
        tests.append({"name": n, "raw_of": t.get("raw_of", n), "label": t["label"], "what": t["what"], "credits": t["credits"],
                      "raw": miss(R), "conv": miss(C), "chi": ph(C, "χ"), "s": ph(C, "s"), "h": ph(C, "h")})
        for x in sents:
            x["px"][n] = [miss([r for r in R if r["file"] == x["id"]]), miss([r for r in C if r["file"] == x["id"]])]
    spent = sum(t["credits"] for t in lab["tests"]) + lab.get("other_credits", 0)
    json.dump({"tests": tests, "sents": sents, "journal": lab["journal"][::-1], "spent": spent,
               "budget": lab["credits_budget"]}, open(page.parent / "lab-data.json", "w"), ensure_ascii=False)
    print(len(tests), "tests;", spent, "credits")


if __name__ == "__main__":
    main()
