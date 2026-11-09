#!/usr/bin/env python3
"""Evaluation for the Chimera -> natural speech-to-speech converter (Phase 4): the corpus v2 validation clips (those
the gate passed, final-v2/val_list.txt) in several versions, each scored on the clips' own phone spans (the timing
survives conversion, so the corpus timing files apply to every version).

    python pie/sts_eval.py prep EVAL                      link the baselines: chimera (the source renders),
                                                          eleven (the paid conversion), restored (the training target)
    python pie/sts_eval.py score EVAL VERSION [VERSION ...] [--tags ZAZY,VALERIA,ANTONIO,W02] [--no-mos]

EVAL/<version>/<TAG>/<clip>__<JD3-PIE|W02-PIE>.wav. Per version and voice: PhoneticXeus broad misses overall and per
phone (pie/gop.py through pie/pilot_score.gop), the acoustic measures of pie/acoustic_map.py (murmur, trill contacts,
bursts, kʷ F2 rise, s centroid), UTMOS (pie/mos.py), and the mel-cepstral distance to the restored target. Writes
EVAL/scores-<version>.json and prints a table.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = Path.home() / "AI/chimera-pie-data"
FULL = DATA / "full"
TAGS = ["ZAZY", "VALERIA", "ANTONIO", "W02"]
SPK = {"ZAZY": "JD3-PIE", "ANTONIO": "JD3-PIE", "VALERIA": "W02-PIE", "W02": "W02-PIE"}


def val_stems():
    """{TAG: [clip__SPK]} for the gate-passed validation clips"""
    out = {t: [] for t in TAGS}
    for l in open(FULL / "final-v2/val_list.txt", encoding="utf-8"):
        f = l.split("|")[0]
        clip, tag = f[:-4].rsplit("__", 1)
        out[tag].append(f"{clip}__{SPK[tag]}")
    return out


def prep(ev):
    stems = val_stems()
    srcs = {"chimera": lambda t, s: FULL / "chimera-v2/renders" / f"{s}.wav",
            "eleven": lambda t, s: FULL / "eleven-v2" / t / f"{s}.wav",
            "restored": lambda t, s: FULL / "restored-v2" / t / f"{s}.wav"}
    for v, f in srcs.items():
        for t, ss in stems.items():
            d = ev / v / t
            d.mkdir(parents=True, exist_ok=True)
            for s in ss:
                p = d / f"{s}.wav"
                if not p.exists():
                    p.symlink_to(f(t, s))
    json.dump(stems, open(ev / "stems.json", "w"), indent=1)
    print({t: len(s) for t, s in stems.items()})


def mcd(a, b, sr):
    import librosa
    n = min(len(a), len(b))
    ca = librosa.feature.mfcc(y=a[:n], sr=sr, n_mfcc=25, n_fft=2048, hop_length=512)[1:]
    cb = librosa.feature.mfcc(y=b[:n], sr=sr, n_mfcc=25, n_fft=2048, hop_length=512)[1:]
    m = min(ca.shape[1], cb.shape[1])
    return float((10 / np.log(10)) * np.sqrt(2) * np.mean(np.linalg.norm(ca[:, :m] - cb[:, :m], axis=0)) / 10)


def score(ev, versions, tags, mos):
    import soundfile as sf
    import pilot_score as PS, acoustic_map as AM, eval as E
    stems = json.load(open(ev / "stems.json"))
    ud = PS.FULLROOT / "runs" / "rp-utts" / "wav"          # the utterance JSONs pie/mos.py reads
    ud.mkdir(parents=True, exist_ok=True)
    for t in tags:
        for s in stems[t]:
            if not (ud / f"{s}.json").exists():
                json.dump(E.utterances(FULL / "timing" / f"{s}.json"), open(ud / f"{s}.json", "w"), ensure_ascii=False)
    for v in versions:
        res = {}
        for t in tags:
            wd = ev / v / t
            if not wd.exists():
                continue
            ss = [s for s in stems[t] if (wd / f"{s}.wav").exists()]
            r = PS.gop(f"sts-{ev.name}-{v}-{t}", wd, ss)
            acc, md = None, []
            for s in ss:
                m = AM.measure_file(wd / f"{s}.wav", AM.phones(FULL / "timing" / f"{s}.json"), t in ("ZAZY", "ANTONIO"))
                acc = {k: list(x) for k, x in m.items()} if acc is None else {k: acc[k] + list(m[k]) for k in acc}
                y, sr = sf.read(wd / f"{s}.wav", dtype="float64"); z, _ = sf.read(FULL / "restored-v2" / t / f"{s}.wav", dtype="float64")
                md.append(mcd(y, z, sr))
            r["acoustic"] = AM.summarize({"x": acc})["x"]
            r["mcd_to_restored"] = round(float(np.mean(md)), 3)
            r["utmos"] = PS.utmos(wd, f"sts-{ev.name}-{v}-{t}") if mos else None
            r["n"] = len(ss)
            res[t] = r
            per = r["per"]
            print(f"{v:14s} {t:8s} n {len(ss):3d}  miss {r['miss']:5.1f} %  kʷ {per.get('kʷ', [0, 0])[0]}/{per.get('kʷ', [0, 0])[1]}"
                  f"  e {per.get('e', [0, 0])[0]}/{per.get('e', [0, 0])[1]}  MCD {r['mcd_to_restored']:.2f}  UTMOS {r['utmos']}", flush=True)
        json.dump(res, open(ev / f"scores-{v}.json", "w"), indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "score"])
    ap.add_argument("eval")
    ap.add_argument("versions", nargs="*")
    ap.add_argument("--tags", default=",".join(TAGS))
    ap.add_argument("--no-mos", action="store_true")
    a = ap.parse_args()
    ev = Path(a.eval)
    ev.mkdir(parents=True, exist_ok=True)
    if a.mode == "prep":
        prep(ev)
    else:
        score(ev, a.versions, a.tags.split(","), not a.no_mos)


if __name__ == "__main__":
    main()
