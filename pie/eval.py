#!/usr/bin/env python3
"""PIE through CHIMERA: render a set of pie_vtl timing files and score them with a phoneme recognizer.

    python pie/eval.py RUN_NAME [--set dev|test|passages|all] [--jobs 12] [--vtl] [--extra "args for the example"]

Sets (DATA = $CHIMERA_PIE_DATA, default ~/AI/chimera-pie-data):
  passages  the dawn hymn and the hard test, male and female (pie_vtl v10 renders)
  dev/test  corpus sentences (pie_vtl v10.1 renders), split alternately by sorted name
Each utterance (breath group) of each file is rendered with the `pie` example on exactly the VTL phone timings,
cut out, and recognized with wav2vec2-xlsr-53-espeak-cv-ft (greedy CTC, CPU). Scores:
  coarse  phone error with near sounds folded (all h-like sounds, palatal = velar, r-like sounds, lax = tense)
  fine    only vowel quality and r-like sounds folded; kʷ counts as k + w, h / χ / ʁ and c / ɟ stay distinct
Writes runs/RUN_NAME/{wav/, scores.json} and prints totals and the target phones most often missed.
--vtl scores the VTL renders themselves instead (the baseline every run is compared against).
"""
import argparse, glob, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = Path(os.environ.get("CHIMERA_PIE_DATA", Path.home() / "AI/chimera-pie-data"))
BIN = ROOT / "target/release/examples/pie"
GAP = 0.12
VOW = set("aeiouə")
MODS = "ʷʱʰ̩̥ːˈ́"


def base(ph):
    return "".join(c for c in ph if c not in MODS)


def utterances(vtl_json):
    """VTL phones -> breath groups of segments with exact durations (as the `pie` example reads them)."""
    d = json.load(open(vtl_json))
    phones = [(p, i == 0, i == len(w["phones"]) - 1) for w in d["words"] for i, p in enumerate(w["phones"])]
    utts, cur = [], None
    for p, first, last in phones:
        if cur is None or p["start"] - cur["end"] >= GAP:
            cur = {"start": p["start"], "end": p["start"], "segs": []}
            utts.append(cur)
        elif cur["segs"] and p["start"] > cur["end"]:
            cur["segs"][-1]["dur"] += p["start"] - cur["end"]
        b = {"ɡ": "g", "ɣ": "ʁ"}.get(base(p["phone"]), base(p["phone"]))
        if b == "r":
            b = "ɹ" if p["kind"] == "Rs" else ("r" if first else "ɾ")
        cur["segs"].append({
            "ipa": b, "dur": round(p["end"] - p["start"], 4), "pie": p["phone"],
            "stress": 1 if (base(p["phone"]) in VOW and p["accent"]) else 0,
            "labial": p["labial"], "breathy": p["breathy"], "long": p.get("long", False),
            "syllabic": p["kind"] in ("Ns", "Ls", "Rs"), "accent": p["accent"],
            "word_end": last, "word_start": first,
        })
        cur["end"] = p["end"]
    return {"duration": d["duration"], "utterances": utts}


# ------------------------------------------------------------------ scoring
COARSE = {'ɡ': 'g', 'c': 'k', 'ɟ': 'g', 'χ': 'h', 'ʁ': 'h', 'ɣ': 'h', 'x': 'h', 'ɦ': 'h', 'ç': 'h', 'ɾ': 'r', 'ɹ': 'r',
          'ʀ': 'r', 'ɐ': 'a', 'ɑ': 'a', 'ɛ': 'e', 'ɔ': 'o', 'ɪ': 'i', 'ʊ': 'u', 'ɜ': 'ə', 'ʌ': 'ə', 'ɵ': 'o', 'æ': 'a',
          'ŋ': 'n', 'ɫ': 'l', 'β': 'b', 'ð': 'd', 'θ': 't', 'ʃ': 's', 'z': 's', 'ʒ': 's', 'j': 'i', 'w': 'u', 'ɲ': 'n',
          'ʔ': '', 'ɒ': 'o', 'ø': 'e', 'y': 'i', 'ɯ': 'u', 'ɨ': 'i', 'ɚ': 'ə', 'ɝ': 'ə'}
FINE = {'ɡ': 'g', 'ɾ': 'r', 'ɹ': 'r', 'ʀ': 'r', 'ɐ': 'a', 'ɑ': 'a', 'ɛ': 'e', 'ɔ': 'o', 'ɪ': 'i', 'ʊ': 'u', 'ɜ': 'ə',
        'ʌ': 'ə', 'ɵ': 'o', 'æ': 'a', 'ɫ': 'l', 'ɒ': 'o', 'ʔ': '', 'x': 'χ', 'ɣ': 'ʁ', 'ɦ': 'h', 'ɚ': 'ə', 'ɝ': 'ə',
        'ɲ': 'n', 'ŋ': 'n', 'ʝ': 'j', 'ç': 'c'}


def norm(phones, table):
    out = []
    for p in phones:
        w = 'ʷ' in p
        for ch in p:
            if ch.isalpha() and ch not in 'ʷʱʰʲː':
                m = table.get(ch, ch)
                if m:
                    out.append(m)
        if w and table is FINE:
            out.append('w')
    return out


def target_seq(segs, table):
    """Target phones with each phone's index (for the per-phone report)."""
    seq, owner = [], []
    for k, s in enumerate(segs):
        b = s["ipa"]
        ph = norm([b], table)
        if table is FINE and s["labial"]:
            ph = ph + ['w']
        seq += ph
        owner += [k] * len(ph)
    return seq, owner


def align(rec, tgt):
    """Edit distance with backtrace; returns (errors, per target index: True if matched)."""
    n, m = len(rec), len(tgt)
    D = np.zeros((n + 1, m + 1), int)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (rec[i - 1] != tgt[j - 1]))
    ok = [False] * m
    got = [None] * m
    i, j = n, m
    while i > 0 and j > 0:
        if D[i, j] == D[i - 1, j - 1] + (rec[i - 1] != tgt[j - 1]):
            ok[j - 1] = rec[i - 1] == tgt[j - 1]
            got[j - 1] = rec[i - 1]
            i, j = i - 1, j - 1
        elif D[i, j] == D[i, j - 1] + 1:
            j -= 1
        else:
            i -= 1
    return int(D[n, m]), ok, got


def sets(name):
    male = sorted(glob.glob(str(DATA / "corpus/*JD3*.json")))
    female = sorted(glob.glob(str(DATA / "corpus/*W02*.json")))
    pas = sorted(glob.glob(str(DATA / "passages/*.json")))
    if name == "passages":
        return pas
    if name == "dev":
        return male[0::2] + female[0::2]
    if name == "test":
        return male[1::2] + female[1::2]
    if name == "devall":
        return pas + male[0::2] + female[0::2]
    return pas + male + female


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--set", default="devall")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--vtl", action="store_true")
    ap.add_argument("--extra", default="")
    ap.add_argument("--wavdir", default="", help="score these WAVs (same names, on the VTL timeline) instead")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = DATA / "runs" / a.run
    (out / "wav").mkdir(parents=True, exist_ok=True)
    files = sets(a.set)
    if a.limit:
        files = files[:a.limit]
    jobs = []
    for f in files:
        stem = Path(f).stem
        body = "female" if "W02" in stem else "male"
        u = utterances(f)
        uj = out / "wav" / f"{stem}.json"
        json.dump(u, open(uj, "w"), ensure_ascii=False)
        wav = Path(a.wavdir) / f"{stem}.wav" if a.wavdir else Path(f).with_suffix(".wav") if a.vtl else out / "wav" / f"{stem}.wav"
        jobs.append((stem, body, u, uj, wav))
    if a.wavdir:
        a.vtl = True
        jobs = [j for j in jobs if j[4].exists()]
    if not a.vtl:
        def render(j):
            stem, body, u, uj, wav = j
            r = subprocess.run([str(BIN), str(uj), str(wav), body] + a.extra.split(), capture_output=True, text=True)
            if r.returncode:
                raise SystemExit(f"{stem}: {r.stderr[-2000:]}")
        with ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(render, jobs))
    sys.path.insert(0, str(Path.home() / "AI/PIE/tts/vc"))
    import librosa
    import recognize as R
    off = 0.0 if a.vtl else 0.3
    tot = {"coarse": [0, 0], "fine": [0, 0]}
    by_set = {}
    per_phone = {}
    items = []
    for stem, body, u, uj, wav in jobs:
        x, _ = librosa.load(str(wav), sr=16000)
        grp = "passage" if not stem.startswith("t") else ("corpus-" + body)
        for k, ut in enumerate(u["utterances"]):
            seg = x[max(0, int((ut["start"] + off - 0.05) * 16000)):int((ut["end"] + off + 0.12) * 16000)]
            if len(seg) < 1600:
                continue
            p, m = R.model()
            import torch
            with torch.no_grad():
                lg = m(p(seg, sampling_rate=16000, return_tensors="pt").input_values).logits[0].numpy()
            rec = R.decode(lg)
            row = {"file": stem, "utt": k, "rec": "".join(rec)}
            for name, table in (("coarse", COARSE), ("fine", FINE)):
                tgt, owner = target_seq(ut["segs"], table)
                e, ok, got = align(norm(rec, table), tgt)
                tot[name][0] += e
                tot[name][1] += len(tgt)
                bs = by_set.setdefault(grp, {"coarse": [0, 0], "fine": [0, 0]})
                bs[name][0] += e
                bs[name][1] += len(tgt)
                row[name] = e
                row["n_" + name] = len(tgt)
                if name == "fine":
                    for t, o, g, ow in zip(tgt, ok, got, owner):
                        key = ut["segs"][ow]["pie"].replace("ˈ", "")
                        key = base(key) + ("ʷ" if "ʷ" in key else "") + ("ʱ" if "ʱ" in key else "") + ("̩" if ut["segs"][ow]["syllabic"] else "")
                        pp = per_phone.setdefault(key, {"n": 0, "miss": 0, "as": {}})
                        pp["n"] += 1
                        if not o:
                            pp["miss"] += 1
                            pp["as"][g or "∅"] = pp["as"].get(g or "∅", 0) + 1
            row["target"] = "".join(target_seq(ut["segs"], FINE)[0])
            items.append(row)
    res = {"run": a.run, "set": a.set, "extra": a.extra, "vtl": a.vtl,
           "coarse": tot["coarse"][0] / max(1, tot["coarse"][1]), "fine": tot["fine"][0] / max(1, tot["fine"][1]),
           "by_set": {g: {n: v[n][0] / max(1, v[n][1]) for n in v} for g, v in by_set.items()},
           "per_phone": per_phone, "items": items}
    json.dump(res, open(out / "scores.json", "w"), ensure_ascii=False, indent=1)
    print(f"{a.run} [{a.set}] coarse {res['coarse']*100:.1f}%  fine {res['fine']*100:.1f}%  " +
          "  ".join(f"{g} {v['fine']*100:.1f}" for g, v in sorted(res["by_set"].items())))
    worst = sorted(per_phone.items(), key=lambda kv: -kv[1]["miss"])[:16]
    print("most missed (fine): " + ", ".join(
        f"{k} {v['miss']}/{v['n']}" + ("(" + ",".join(f"{g}{c}" for g, c in sorted(v['as'].items(), key=lambda x: -x[1])[:3]) + ")")
        for k, v in worst))


if __name__ == "__main__":
    main()
