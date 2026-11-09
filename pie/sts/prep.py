"""Build the converter's training pairs: for every gate-passed clip of corpus v2 (train and val), the source render's
and the restored target's BigVGAN log-mels, frame-aligned, as float16 in DATA/<split>/<TAG>__<stem>.npz, after an
alignment check (lag of the best envelope correlation, 5 ms frames, +-100 ms; pairs off by more than 5 ms are dropped).
The target is cut to the source's length (the conversion pads its end to a 2048-sample block).
    python pie/sts/prep.py [--workers 16]"""
import argparse, json, multiprocessing as mp, sys
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def lag(a, b, f=220):
    def env(x):
        m = len(x) // f
        return np.abs(x[:m * f]).reshape(m, f).mean(1)
    ea, eb = env(a), env(b)
    ks = range(-20, 21)
    c = [np.corrcoef(ea[max(0, k):len(ea) + min(0, k)], eb[max(0, -k):len(eb) - max(0, k)])[0, 1] for k in ks]
    return list(ks)[int(np.nanargmax(c))] * f / C.SR


def one(job):
    split, tag, stem = job
    out = C.DATA / split / f"{tag}__{stem}.npz"
    if out.exists():
        return {"split": split, "tag": tag, "stem": stem, "skipped": True}
    s, sr = sf.read(C.RENDERS / f"{stem}.wav", dtype="float32"); t, sr2 = sf.read(C.RESTORED / tag / f"{stem}.wav", dtype="float32")
    assert sr == sr2 == C.SR
    if len(t) < len(s):
        t = np.pad(t, (0, len(s) - len(t)))
    t = t[:len(s)]
    lg = lag(s, t)
    r = {"split": split, "tag": tag, "stem": stem, "samples": len(s), "lag_s": round(lg, 4)}
    if abs(lg) > 0.005:
        r["dropped"] = "lag"
        return r
    ms, mt = C.mel(s).numpy(), C.mel(t).numpy()
    n = min(ms.shape[1], mt.shape[1])
    np.savez(out, src=ms[:, :n].astype(np.float16), tgt=mt[:, :n].astype(np.float16))
    r["frames"] = n
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=16); a = ap.parse_args()
    jobs = [(sp, t, s) for sp in ("train", "val") for t, s in C.pairs(sp)]
    for sp in ("train", "val"):
        (C.DATA / sp).mkdir(parents=True, exist_ok=True)
    import torch; torch.set_num_threads(1)
    with mp.Pool(a.workers) as p, open(C.DATA / "manifest.jsonl", "a") as man:
        rows = []
        for r in p.imap_unordered(one, jobs, chunksize=8):
            if not r.get("skipped"):
                man.write(json.dumps(r) + "\n"); rows.append(r)
    lags = [r["lag_s"] for r in rows if "lag_s" in r]
    print(len(jobs), "pairs;", sum("dropped" in r for r in rows), "dropped for lag;",
          "lags (s):", {k: lags.count(k) for k in sorted(set(lags))}, flush=True)


if __name__ == "__main__":
    main()
