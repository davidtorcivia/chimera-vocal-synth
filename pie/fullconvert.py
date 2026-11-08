#!/usr/bin/env python3
"""Convert the Chimera corpus renders with ElevenLabs speech-to-speech (the corpus settings of the PIE repo's
tts/vtl/corpus_pipeline.py), voice by voice, resumably, under a hard cap on converted minutes.

    python pie/fullconvert.py PLAN.json [--workers 8] [--cap-min 1050] [--limit N]
PLAN.json: {"renders": DIR, "out": DIR, "voices": [{"tag": "ZAZY", "voice_id": ..., "speaker": "JD3-PIE",
            "clips": "all" | FILE_OF_STEMS}, ...]}
Writes OUT/<tag>/<CLIP>__<SPEAKER>.wav (44.1 kHz 16-bit, ElevenLabs pcm_44100) and OUT/log.jsonl (one line per
request: tag, stem, cost, request id, seconds, or error). Skips files already converted. Stops at once on 401/402/403
(auth, quota, payment) or when the converted minutes reach --cap-min; any other error is logged and the clip retried
on the next run.
"""
import argparse, json, sys, threading, time, wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path.home() / "AI/PIE/tts/vtl"))
import corpus_pipeline as cp


def seconds(path):
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cap-min", type=float, default=1050.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    rd, out = Path(plan["renders"]), Path(plan["out"])
    out.mkdir(parents=True, exist_ok=True)
    key = (Path.home() / ".config/elevenlabs/api_key").read_text().strip()
    log_path = out / "log.jsonl"
    done_min = 0.0
    if log_path.exists():
        for l in open(log_path):
            r = json.loads(l)
            if "error" not in r:
                done_min += r.get("seconds", 0) / 60
    jobs = []
    for v in plan["voices"]:
        cp.VOICES[v["tag"]] = v["voice_id"]
        (out / v["tag"]).mkdir(exist_ok=True)
        if v["clips"] == "all":
            stems = sorted(p.stem for p in rd.glob(f"*__{v['speaker']}.wav"))
        else:
            stems = open(v["clips"]).read().split()
        for s in stems:
            if not (out / v["tag"] / f"{s}.wav").exists():
                jobs.append((v["tag"], s))
    if a.limit:
        jobs = jobs[: a.limit]
    todo_min = sum(seconds(rd / f"{s}.wav") for _, s in jobs) / 60
    print(f"{len(jobs)} conversions to do, {todo_min:.0f} min; {done_min:.0f} min already converted; cap {a.cap_min:.0f} min", flush=True)
    stop = threading.Event()
    lock = threading.Lock()
    stats = {"n": 0, "cost": 0, "sec": 0.0, "err": 0}

    def one(job):
        tag, s = job
        if stop.is_set():
            return None
        with lock:
            if done_min + stats["sec"] / 60 >= a.cap_min:
                stop.set()
                return None
        m = {"vtl_wav": str(rd / f"{s}.wav"), "eleven_wav": str(out / tag / f"{s}.wav"), "speaker": tag}
        r = cp._convert(Path("/"), m, key)
        r = {"tag": tag, "stem": s, **r}
        if "error" not in r:
            src = seconds(rd / f"{s}.wav")
            if abs(r["seconds"] - src) > 0.05:
                r["length_note"] = f"source {src:.3f} s"
        return r

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex, open(log_path, "a", encoding="utf-8") as lg:
        futs = [ex.submit(one, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            if r is None:
                continue
            lg.write(json.dumps(r, ensure_ascii=False) + "\n")
            lg.flush()
            with lock:
                stats["n"] += 1
                if "error" in r:
                    stats["err"] += 1
                    print("ERROR", r, flush=True)
                    if r["error"] in (401, 402, 403):
                        stop.set()
                else:
                    stats["cost"] += r["cost"]
                    stats["sec"] += r["seconds"]
                if stats["n"] % 100 == 0:
                    el = time.time() - t0
                    print(f"progress {stats['n']}/{len(jobs)}  {stats['sec'] / 60:.0f} min  {stats['cost']} credits  "
                          f"{stats['err']} errors  {el / 60:.0f} min elapsed", flush=True)
    print(f"done: {stats['n']} conversions, {stats['sec'] / 60:.1f} min, {stats['cost']} credits, {stats['err']} errors"
          + ("  (STOPPED: cap or billing)" if stop.is_set() else ""), flush=True)


if __name__ == "__main__":
    main()
