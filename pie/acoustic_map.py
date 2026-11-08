#!/usr/bin/env python3
"""Acoustic damage map for corpus v2: the sounds a phone recognizer cannot judge, measured inside each phone's span
in the Chimera source and in its conversion (the two are time-aligned).

    python pie/acoustic_map.py OUT.json [--n 300] [--procs 8]
Per voice (ZAZY ANTONIO VALERIA W02), on N random clips:
  murmur   H1*-H2* (dB, Iseli-Alwan) and CPPS on the first third of the vowel after bʰ dʰ gʰ ɟʰ gʷʰ, minus the same
           after plain voiced stops (b d g ɟ gʷ): breathy voice raises H1*-H2* and lowers CPPS
  trill    contacts (intensity dips ≥ 6 dB, 8 ms window) per non-syllabic r; the window covers the 110 ms Chimera
           gives the trill (centre ± 55 ms) when both neighbours are vowels, else the span ± 10 ms
  h        level re the following vowel (dB) and 2-8 kHz share
  burst    p t k: 2-8 kHz energy in the 15 ms after the release re the following vowel (dB); VOT (ms)
  round    kʷ gʷ vs k g: F2 rise from 15 ms into the following vowel to its middle (Hz); rounding lowers onset F2
  s        spectral centroid 2-11 kHz (Hz): a slide toward ʃ lowers it
  vowels   u, i, e, o: level re the clip's median vowel (dB) and F2
  ltas     the voice's long-term spectrum in bands (dB re 1-4 kHz)
"""
import argparse, json, random, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval as E
import murmur as MU

FULL = Path.home() / "AI/chimera-pie-data/full"
VOICES = {"ZAZY": "JD3-PIE", "ANTONIO": "JD3-PIE", "VALERIA": "W02-PIE", "W02": "W02-PIE"}
VOW = set("aeiouə")
STOPS = {"b", "d", "g", "ɟ"}


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9) if len(x) else float("nan")


def band(x, sr, lo, hi):
    if len(x) < 32:
        return float("nan")
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return float(X[(f >= lo) & (f < hi)].sum())


def centroid(x, sr, lo=2000, hi=11000):
    if len(x) < 64:
        return float("nan")
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    m = (f >= lo) & (f < hi)
    return float((X[m] * f[m]).sum() / (X[m].sum() + 1e-20))


def phones(timing):
    d = json.load(open(timing))
    out = []
    for w in d["words"]:
        for p in w["phones"]:
            b = E.base(p["phone"])
            b = {"ɡ": "g", "ɣ": "ʁ"}.get(b, b)
            out.append({"b": b, "t0": p["start"], "t1": p["end"], "lab": p["labial"], "br": p["breathy"],
                        "syl": p["kind"] in ("Ns", "Ls", "Rs")})
    return out


def measure_file(wav, ph, male):
    snd = parselmouth.Sound(str(wav))
    x = snd.values[0]
    sr = snd.sampling_frequency
    seg = lambda a, b: x[max(0, int(a * sr)):max(0, int(b * sr))]
    inten = snd.to_intensity(minimum_pitch=400, time_step=0.001)
    fm = snd.to_formant_burg(time_step=0.004, max_number_of_formants=5, maximum_formant=5000 if male else 5500)
    pitch = snd.to_pitch(time_step=0.002, pitch_floor=60 if male else 100, pitch_ceiling=300 if male else 450)
    R = {k: [] for k in ("mur", "mod", "trill", "h_lvl", "h_share", "burst", "vot", "rnd_lab", "rnd_pl", "s_cent", "vlev", "vf2")}
    vls = [rms_db(seg(p["t0"] + 0.25 * (p["t1"] - p["t0"]), p["t1"] - 0.25 * (p["t1"] - p["t0"]))) for p in ph if p["b"][:1] in VOW and len(p["b"]) == 1]
    vmed = float(np.nanmedian(vls)) if vls else float("nan")
    for i, p in enumerate(ph):
        nx = ph[i + 1] if i + 1 < len(ph) else None
        pv = ph[i - 1] if i > 0 else None
        nv = nx is not None and nx["b"][:1] in VOW and abs(nx["t0"] - p["t1"]) < 0.02
        b = p["b"]
        # murmur vs modal: the first third of the vowel after a voiced stop
        if nv and b in STOPS:
            dv = nx["t1"] - nx["t0"]
            t0, t1 = nx["t0"], nx["t0"] + max(0.03, dv / 3)
            try:
                h, c, v = MU.measure(snd, t0, t1)
                if np.isfinite(h) and v > 0.5:
                    R["mur" if p["br"] else "mod"].append((h, c))
            except Exception:
                pass
        if b == "r" and not p["syl"]:
            both = pv is not None and pv["b"][:1] in VOW and nv
            mid = 0.5 * (p["t0"] + p["t1"])
            a0, a1 = (mid - 0.055, mid + 0.055) if both else (p["t0"] - 0.01, p["t1"] + 0.01)
            ts = np.arange(a0, a1, 0.001)
            v = np.array([inten.get_value(t) or np.nan for t in ts])
            v = v[np.isfinite(v)]
            n = 0
            if len(v) > 10:
                k = 1
                while k < len(v) - 1:
                    if v[k] < v[k - 1] and v[k] <= v[k + 1]:
                        left, right = v[max(0, k - 15):k].max(), v[k + 1:k + 16].max()
                        if min(left, right) - v[k] >= 6:
                            n += 1
                            k += 8
                    k += 1
            R["trill"].append(n)
        if b == "h" and nv:
            hs, vs = seg(p["t0"], p["t1"]), seg(nx["t0"], nx["t1"])
            R["h_lvl"].append(rms_db(hs) - rms_db(vs))
            R["h_share"].append(band(hs, sr, 2000, 8000) / (band(hs, sr, 50, 8000) + 1e-20))
        if b in ("p", "t", "k") and not p["lab"] and nv:
            rel = p["t1"]
            bu, vs = seg(rel, rel + 0.015), seg(nx["t0"] + 0.03, nx["t1"])
            R["burst"].append(10 * np.log10(band(bu, sr, 2000, 8000) / len(bu) + 1e-20) - 10 * np.log10(band(vs, sr, 2000, 8000) / max(1, len(vs)) + 1e-20))
            ts = np.arange(rel, min(nx["t1"], rel + 0.08), 0.002)
            f0 = [pitch.get_value_at_time(t) for t in ts]
            on = next((t for t, f in zip(ts, f0) if f and f > 0), None)
            if on is not None:
                R["vot"].append((on - rel) * 1000)
        if b in ("k", "g") and nv and nx["b"] in ("a", "e", "i"):
            a = fm.get_value_at_time(2, nx["t0"] + 0.015)
            m = fm.get_value_at_time(2, 0.5 * (nx["t0"] + nx["t1"]))
            if a and m:
                R["rnd_lab" if p["lab"] else "rnd_pl"].append(m - a)
        if b == "s":
            R["s_cent"].append(centroid(seg(p["t0"] + 0.01, p["t1"] - 0.01), sr))
        if b in ("u", "i", "e", "o") and len(b) == 1:
            d = p["t1"] - p["t0"]
            R["vlev"].append((b, rms_db(seg(p["t0"] + 0.25 * d, p["t1"] - 0.25 * d)) - vmed))
            f2 = fm.get_value_at_time(2, 0.5 * (p["t0"] + p["t1"]))
            if f2:
                R["vf2"].append((b, f2))
    return R


def ltas(wav):
    snd = parselmouth.Sound(str(wav))
    x = snd.values[0]
    sr = snd.sampling_frequency
    n = int(0.04 * sr)
    fr = x[: len(x) // n * n].reshape(-1, n)
    rms = np.sqrt((fr ** 2).mean(1))
    sp = fr[rms > np.percentile(rms, 60)]
    S = (np.abs(np.fft.rfft(sp * np.hanning(n), axis=1)) ** 2).mean(0)
    f = np.fft.rfftfreq(n, 1 / sr)
    return f, S


def job(args):
    tag, stem = args
    timing = FULL / "timing" / f"{stem}.json"
    ph = phones(timing)
    male = VOICES[tag] == "JD3-PIE"
    out = {}
    for kind, wav in (("src", FULL / "chimera-v2/renders" / f"{stem}.wav"), ("conv", FULL / "eleven-v2" / tag / f"{stem}.wav")):
        out[kind] = measure_file(wav, ph, male)
    return tag, out


def summarize(acc):
    s = {}
    for kind, R in acc.items():
        m = np.array(R["mur"]) if R["mur"] else np.zeros((0, 2))
        o = np.array(R["mod"]) if R["mod"] else np.zeros((0, 2))
        tr = np.array(R["trill"])
        vl = {v: float(np.nanmedian([d for b, d in R["vlev"] if b == v])) for v in "uieo"}
        vf = {v: float(np.nanmedian([d for b, d in R["vf2"] if b == v])) for v in "uieo"}
        s[kind] = {
            "murmur_h1h2_minus_modal": float(np.nanmean(m[:, 0]) - np.nanmean(o[:, 0])) if len(m) and len(o) else None,
            "murmur_cpps_minus_modal": float(np.nanmean(m[:, 1]) - np.nanmean(o[:, 1])) if len(m) and len(o) else None,
            "n_murmur": len(m),
            "trill_2plus": float(np.mean(tr >= 2)) if len(tr) else None, "trill_0": float(np.mean(tr == 0)) if len(tr) else None,
            "trill_median": float(np.median(tr)) if len(tr) else None, "n_trill": len(tr),
            "h_level_db": float(np.nanmedian(R["h_lvl"])), "h_share": float(np.nanmedian(R["h_share"])),
            "burst_db": float(np.nanmedian(R["burst"])), "vot_ms": float(np.nanmedian(R["vot"])) if R["vot"] else None,
            "kw_f2_rise": float(np.nanmedian(R["rnd_lab"])) if R["rnd_lab"] else None,
            "k_f2_rise": float(np.nanmedian(R["rnd_pl"])) if R["rnd_pl"] else None,
            "s_centroid": float(np.nanmedian(R["s_cent"])),
            "vowel_level_db": vl, "vowel_f2": vf,
        }
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    jobs = []
    for tag in VOICES:
        stems = sorted(p.stem for p in (FULL / "eleven-v2" / tag).glob("*.wav"))
        random.Random(f"amap-{tag}").shuffle(stems)
        jobs += [(tag, s) for s in stems[: a.n]]
    acc = {t: {"src": None, "conv": None} for t in VOICES}
    with ProcessPoolExecutor(a.procs) as ex:
        for tag, out in ex.map(job, jobs, chunksize=4):
            for kind in ("src", "conv"):
                if acc[tag][kind] is None:
                    acc[tag][kind] = {k: [] for k in out[kind]}
                for k, v in out[kind].items():
                    acc[tag][kind][k].extend(v)
    res = {t: summarize(acc[t]) for t in VOICES}
    # long-term spectra of each voice's converted audio (first 60 sampled clips), in bands re 1-4 kHz
    for t in VOICES:
        stems = [s for tg, s in jobs if tg == t][:60]
        S = None
        for s in stems:
            f, Sx = ltas(FULL / "eleven-v2" / t / f"{s}.wav")
            S = Sx if S is None else S + Sx
        e = lambda lo, hi: S[(f >= lo) & (f < hi)].sum()
        ref = e(1000, 4000)
        res[t]["ltas"] = {f"{lo}-{hi}": round(float(10 * np.log10(e(lo, hi) / ref)), 1) for lo, hi in
                          [(80, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 12000), (12000, 16000)]}
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
