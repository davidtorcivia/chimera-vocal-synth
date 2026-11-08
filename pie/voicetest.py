#!/usr/bin/env python3
"""Convert the coverage passage (every PIE sound ≥ 4 times, ~25 s) into each candidate voice and score it.

    python pie/voicetest.py VOICES.json [--cap 7000] [--workers 4]
VOICES.json: [{"name": ..., "voice_id": ..., "gender": "male"|"female", "lang": ...}, ...]
For each: DATA/cover/conv/NAME.wav (ElevenLabs speech-to-speech, the corpus settings), its PhoneticXeus score on
the passage timing (pie/gop.py, broad), misses by sound class, and the converted audio's noise/hum/band/air
(pie/voicescreen.measure). Results: DATA/cover/results.jsonl (one line per voice; reruns skip converted voices).
"""
import argparse, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path.home() / "AI/PIE/tts/vtl"))
import eval as E
import voicescreen as V
import corpus_pipeline as cp

DATA = E.DATA
STEM = {"male": "cover-male-JD3", "female": "cover-female-W02"}
CLASSES = {
    "vowels": ["a", "e", "i", "o", "u", "ə"],
    "plain stops": ["p", "b", "t", "d", "k", "g"],
    "murmured": ["bʱ", "dʱ", "gʱ", "ɟʱ", "gʷʱ"],
    "palatal": ["c", "ɟ"],
    "labiovelar": ["kʷ", "gʷ", "gʷʱ"],
    "laryngeal/fric": ["h", "χ", "ʁ", "ʁʷ", "s"],
    "sonorants": ["m", "n", "l", "r", "j", "w"],
    "syllabic": ["m̩", "n̩", "l̩", "r̩"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("voices")
    ap.add_argument("--cap", type=int, default=7000)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    voices = json.load(open(a.voices))
    out = DATA / "cover"
    (out / "conv").mkdir(parents=True, exist_ok=True)
    key = (Path.home() / ".config/elevenlabs/api_key").read_text().strip()
    res_path = out / "results.jsonl"
    done = {json.loads(l)["name"] for l in open(res_path)} if res_path.exists() else set()
    todo = [v for v in voices if v["name"] not in done]
    spent = 0

    def convert(v):
        cp.VOICES[v["name"]] = v["voice_id"]
        stem = STEM[v["gender"]]
        m = {"vtl_wav": f"raw/{stem}.wav", "eleven_wav": f"conv/{v['name']}.wav", "speaker": v["name"]}
        return v, cp._convert(out, m, key)

    with ThreadPoolExecutor(a.workers) as ex:
        convs = list(ex.map(convert, todo))
    with open(res_path, "a", encoding="utf-8") as fo:
        for v, r in convs:
            if "error" in r:
                print("ERROR", v["name"], r)
                continue
            spent += r["cost"]
            stem = STEM[v["gender"]]
            wd = out / "score" / v["name"]
            wd.mkdir(parents=True, exist_ok=True)
            (wd / f"{stem}.wav").unlink(missing_ok=True)
            (wd / f"{stem}.wav").symlink_to(out / "conv" / f"{v['name']}.wav")
            (wd / "stems.txt").write_text(stem + "\n")
            run = f"cv-{v['name']}"
            subprocess.run([sys.executable, "-I", str(HERE / "gop.py"), run, "--wavdir", str(wd), "--set", "passages",
                            "--only", str(wd / "stems.txt"), "--threads", "14"], capture_output=True, text=True)
            g = json.load(open(DATA / "runs" / run / "gop.json"))
            pp = g["per_phone"]
            cls = {}
            for c, ks in CLASSES.items():
                n = sum(pp[k]["n"] for k in ks if k in pp)
                miss = sum(pp[k]["n"] - pp[k]["broad"] for k in ks if k in pp)
                cls[c] = round(100 * miss / n) if n else None
            x, sr = sf.read(out / "conv" / f"{v['name']}.wav", dtype="float32")
            rec = {**v, "cost": r["cost"], "miss": round(100 * (1 - g["broad"]), 1), "classes": cls,
                   "per_phone": {k: [pp[k]["n"] - pp[k]["broad"], pp[k]["n"]] for k in pp}, **V.measure(x, sr)}
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{v['name']:28} {rec['miss']:5.1f}%  " + "  ".join(f"{c} {p}" for c, p in cls.items()), flush=True)
    print("credits", spent)


if __name__ == "__main__":
    main()
