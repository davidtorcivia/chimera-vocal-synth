#!/usr/bin/env python3
"""Murmur by acoustics: formant-corrected H1*-H2* (Iseli & Alwan 2004, as VoiceSauce/PraatSauce) and CPPS on the
first third of the vowel after a murmured stop, against the same vowel after the plain stop.

    python pie/murmur.py [male|female] [option=value ...]
Renders "ábʰa ‖ ába ‖ ádʰa ‖ áda ‖ ágʰa ‖ ága ‖ ébʰe ‖ ébe ‖ ódʰo ‖ ódo" with `chimera render pie`. Breathy voice
raises H1*-H2* over modal by several dB (Keating et al. 2011: the measure that separates phonation types in
every language they studied); CPPS falls. Prints both and the voiced share of the vowel's first third.
"""
import os, subprocess, sys
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / os.environ.get("CHIMERA_BIN", "target/release/chimera")
OUT = Path(os.environ.get("CHIMERA_PIE_DATA", Path.home() / "AI/chimera-pie-data")) / "acoustic" / "murmur"
PAIRS = "ábʰa ‖ ába ‖ ádʰa ‖ áda ‖ ágʰa ‖ ága ‖ ébʰe ‖ ébe ‖ ódʰo ‖ ódo"


def corr(f, F, B, fs):
    """Iseli–Alwan correction (dB) for harmonic frequency f, given formant F with bandwidth B."""
    r = np.exp(-np.pi * B / fs)
    w, wc = 2 * np.pi * f / fs, 2 * np.pi * F / fs
    num = (1 - 2 * r * np.cos(wc) + r * r) ** 2
    den = (1 - 2 * r * np.cos(wc + w) + r * r) * (1 - 2 * r * np.cos(wc - w) + r * r)
    return 10 * np.log10(num / den)


def measure(snd, t0, t1):
    part = snd.extract_part(t0, t1, parselmouth.WindowShape.HAMMING, 1.0, False)
    pitch = part.to_pitch(time_step=0.005, pitch_floor=70, pitch_ceiling=400)
    f0s = pitch.selected_array["frequency"]
    voiced = float(np.mean(f0s > 0)) if len(f0s) else 0.0
    f0 = float(np.median(f0s[f0s > 0])) if np.any(f0s > 0) else float("nan")
    if not np.isfinite(f0):
        return float("nan"), float("nan"), voiced
    spec = part.to_spectrum()
    fr = np.array([spec.get_frequency_from_bin_number(i + 1) for i in range(spec.get_number_of_bins())])
    mag = 20 * np.log10(np.abs(spec.values[0] + 1j * spec.values[1]) + 1e-12)
    amp = lambda h: mag[(fr > h * f0 * 0.85) & (fr < h * f0 * 1.15)].max()
    fm = part.to_formant_burg(time_step=0.005, max_number_of_formants=5, maximum_formant=5000 if f0 < 160 else 5500)
    tm = part.duration / 2
    F1, F2 = fm.get_value_at_time(1, tm), fm.get_value_at_time(2, tm)
    B1, B2 = fm.get_bandwidth_at_time(1, tm), fm.get_bandwidth_at_time(2, tm)
    fs = part.sampling_frequency
    h1 = amp(1) - corr(f0, F1, B1, fs) - corr(f0, F2, B2, fs)
    h2 = amp(2) - corr(2 * f0, F1, B1, fs) - corr(2 * f0, F2, B2, fs)
    cpps = call(call(part, "To PowerCepstrogram", 60, 0.002, 5000, 50), "Get CPPS", False, 0.01, 0.001, 60, 330, 0.05, "parabolic", 0.001, 0, "Straight", "Robust")
    return h1 - h2, cpps, voiced


def main():
    body = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("male", "female") else "male"
    opts = [a for a in sys.argv[1:] if "=" in a]
    r = subprocess.run([str(BIN), "render", "pie", str(OUT), PAIRS, body, *opts, "--float"], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stderr)
    marks = [(t.rsplit("@", 1)[0], float(t.rsplit("@", 1)[1])) for t in [l for l in r.stdout.split("\n") if "@" in l][0].split()]
    snd = parselmouth.Sound(str(OUT) + ".wav")
    res = []
    for k in range(0, len(marks), 3):
        (v1, a), (c, b), (v2, t2) = marks[k:k + 3]
        end = marks[k + 3][1] - 0.3 if k + 3 < len(marks) else t2 + 0.2
        third = (end - t2) / 3
        res.append((c, *measure(snd, t2 + 0.005, t2 + max(0.03, third))))
    for (cm, hm, cpm, vm), (cp, hp, cpp, vp) in zip(res[0::2], res[1::2]):
        print(f"{cm:4} vs {cp:3}  H1*-H2* {hm:6.1f} vs {hp:6.1f}  (Δ {hm - hp:+5.1f} dB)   CPPS {cpm:5.1f} vs {cpp:5.1f}   voiced {vm*100:3.0f}% vs {vp*100:3.0f}%")


if __name__ == "__main__":
    main()
