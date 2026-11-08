#!/usr/bin/env python3
"""Screen ElevenLabs shared-library voices by the quality of their own preview audio.

For each language × gender: the most-cloned professional voices at the standard rate; each preview is measured for
  noise   the quiet frames' level against the speech frames' (10th vs 95th percentile of 20 ms RMS, dB; lower = cleaner)
  hum     mains hum: 50/60 Hz and their first harmonics against the neighbouring bins, in the quiet frames (dB)
  band    the highest frequency where the long-term spectrum is within 50 dB of its peak (Hz; low = dull or band-limited)
  air     energy 6-12 kHz against 0.3-3 kHz (dB; very low = muffled, the "poor spectrum response")
Writes OUT/voices.jsonl and the previews under OUT/previews/.

    python pie/voicescreen.py OUT [--per 12] [--langs hi,mr,es,...]
"""
import argparse, io, json, subprocess, urllib.parse, urllib.request
from pathlib import Path

import numpy as np

KEY = (Path.home() / ".config/elevenlabs/api_key").read_text().strip()


def api(params):
    url = "https://api.elevenlabs.io/v1/shared-voices?" + urllib.parse.urlencode(params)
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"xi-api-key": KEY}), timeout=60))


def decode(path, sr=44100):
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32), sr


def measure(x, sr):
    n = int(0.02 * sr)
    fr = x[: len(x) // n * n].reshape(-1, n)
    rms = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)
    lo, hi = np.percentile(rms, 10), np.percentile(rms, 95)
    quiet = fr[rms <= lo + 3]
    # long-term spectrum of the speech frames
    sp = fr[rms >= hi - 20]
    S = (np.abs(np.fft.rfft(sp * np.hanning(n), axis=1)) ** 2).mean(0)
    f = np.fft.rfftfreq(n, 1 / sr)
    Sd = 10 * np.log10(S + 1e-20)
    # smooth over ~250 Hz
    k = max(1, int(250 / (f[1] - f[0])))
    Sm = np.convolve(Sd, np.ones(k) / k, mode="same")
    peak = Sm[(f > 100) & (f < 5000)].max()
    above = f[(Sm > peak - 50) & (f > 1000)]
    band = float(above.max()) if len(above) else 0.0
    e = lambda a, b: S[(f >= a) & (f < b)].sum()
    air = 10 * np.log10(e(6000, 12000) / (e(300, 3000) + 1e-20) + 1e-20)
    # hum in the quiet frames (longer window for resolution)
    hum = 0.0
    if len(quiet) >= 5:
        q = quiet.reshape(-1)
        Q = np.abs(np.fft.rfft(q * np.hanning(len(q)))) ** 2
        fq = np.fft.rfftfreq(len(q), 1 / sr)
        vals = []
        for h in (50, 60, 100, 120, 150, 180):
            band_ = Q[(fq > h - 2) & (fq < h + 2)].mean()
            nb = Q[((fq > h - 15) & (fq < h - 5)) | ((fq > h + 5) & (fq < h + 15))].mean()
            vals.append(10 * np.log10(band_ / (nb + 1e-20) + 1e-20))
        hum = float(max(vals))
    return {"noise_db": round(float(lo - hi), 1), "hum_db": round(hum, 1), "band_hz": round(band), "air_db": round(float(air), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--per", type=int, default=12)
    ap.add_argument("--langs", default="hi,mr,es,it,sr,hr,cs,de,nl,ar,lt,el")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "previews").mkdir(parents=True, exist_ok=True)
    seen = set()
    with open(out / "voices.jsonl", "w", encoding="utf-8") as fo:
        for lang in a.langs.split(","):
            for gender in ("male", "female"):
                try:
                    d = api({"page_size": 60, "language": lang, "gender": gender, "category": "professional", "sort": "cloned_by_count"})
                except Exception as e:
                    print(lang, gender, "search failed", e)
                    continue
                vs = [v for v in d["voices"] if (v.get("rate") or 1) <= 1 and v.get("free_users_allowed", True) and v["voice_id"] not in seen]
                for v in vs[: a.per]:
                    seen.add(v["voice_id"])
                    p = out / "previews" / f"{v['voice_id']}.mp3"
                    try:
                        if not p.exists():
                            p.write_bytes(urllib.request.urlopen(v["preview_url"], timeout=60).read())
                        m = measure(*decode(p))
                    except Exception as e:
                        m = {"error": str(e)[:120]}
                    rec = {"lang": lang, "gender": gender, "voice_id": v["voice_id"], "name": v["name"], "accent": v.get("accent"),
                           "locale": v.get("locale"), "age": v.get("age"), "use_case": v.get("use_case"), "cloned_by": v.get("cloned_by_count"),
                           "owner": v.get("public_owner_id"), "preview": v["preview_url"], **m}
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(lang, gender, len(vs[: a.per]), "voices", flush=True)


if __name__ == "__main__":
    main()
