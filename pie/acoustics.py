#!/usr/bin/env python3
"""Acoustic checks for the PIE sounds a phoneme recognizer cannot judge.

    chimera render pie OUT "áka ‖ áḱa ‖ ..." male --float > OUT.marks
    python pie/acoustics.py OUT.wav OUT.marks

For each VCV item (V C V, the C's marks give its span) it measures:
  burst    stops: spectral peak and centroid (Hz) of the 12 ms after the release
  F2 on/mid  the following vowel's F2 15 ms after its start and at its middle (LPC)
  noise dB   fricatives: level of the consonant re the vowels (dB) and its spectral centroid (Hz)
  voiced %   share of the consonant with periodic voicing (autocorrelation)
  H1-H2      the following vowel's first 30 ms (breathy voice raises it)
Expected values (adult male, from the phonetics literature) are printed beside the measures.
"""
import sys
import numpy as np
import soundfile as sf
import librosa


def marks(path):
    txt = open(path).read().split("\n")
    line = [l for l in txt if "@" in l][0]
    out = []
    for tok in line.split():
        lab, t = tok.rsplit("@", 1)
        out.append((lab, float(t)))
    return out


def lpc_formants(x, sr, order=None):
    x = x * np.hamming(len(x))
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    order = order or int(2 + sr / 1000)
    a = librosa.lpc(x, order=order)
    r = [z for z in np.roots(a) if np.imag(z) >= 0]
    ang = np.arctan2(np.imag(r), np.real(r))
    f = sorted(ang * sr / (2 * np.pi))
    bw = [-0.5 * sr / np.pi * np.log(abs(z)) for z in r]
    fs = sorted([fi for fi, b in zip(ang * sr / (2 * np.pi), bw) if fi > 150 and b < 500])
    return fs


def spectrum(x, sr):
    w = np.hanning(len(x))
    X = np.abs(np.fft.rfft(x * w, n=4096))
    f = np.fft.rfftfreq(4096, 1 / sr)
    return f, X


def voiced_share(x, sr, f0lo=70, f0hi=400):
    hop, win = int(0.005 * sr), int(0.025 * sr)
    n = v = 0
    for i in range(0, max(1, len(x) - win), hop):
        fr = x[i:i + win]
        if np.sqrt(np.mean(fr ** 2)) < 1e-4:
            n += 1
            continue
        ac = np.correlate(fr, fr, "full")[win - 1:]
        lo, hi = int(sr / f0hi), int(sr / f0lo)
        if hi >= len(ac):
            continue
        r = ac[lo:hi].max() / (ac[0] + 1e-12)
        n += 1
        v += r > 0.5
    return v / max(1, n)


def h1h2(x, sr):
    f0 = librosa.yin(x, fmin=70, fmax=400, sr=sr, frame_length=min(len(x), 2048))
    f0 = float(np.median(f0))
    f, X = spectrum(x, sr)
    def amp(h):
        band = (f > h * f0 * 0.8) & (f < h * f0 * 1.2)
        return 20 * np.log10(X[band].max() + 1e-12)
    return amp(1) - amp(2)


def main():
    wav, mk = sys.argv[1], sys.argv[2]
    x, sr = sf.read(wav, dtype="float32")
    m = marks(mk)
    seg = lambda a, b: x[int(a * sr):int(b * sr)]
    items = []
    # an item starts at its accented vowel (á í ú): V C... V
    starts = [k for k, (lab, _) in enumerate(m) if "\u0301" in lab and lab[0] in "aeiou"] + [len(m)]
    for a, b in zip(starts, starts[1:]):
        g = m[a:b]
        if len(g) < 3:
            continue
        v1, t0 = g[0]
        c = "".join(l for l, _ in g[1:-1])
        t1 = g[1][1]
        v2, t2 = g[-1]
        t3 = m[b][1] - 0.3 if b < len(m) else t2 + 0.2
        t3 = min(max(t3, t2 + 0.1), t2 + 0.25)
        items.append((v1, c, v2, t0, t1, t2, t3))
    print(f"{'item':8} {'burst pk':>8} {'centr':>6} {'F2on':>6} {'F2mid':>6} {'noise':>6} {'ncentr':>6} {'voiced':>6} {'H1-H2':>6}")
    for v1, c, v2, t0, t1, t2, t3 in items:
        vowel_rms = np.sqrt(np.mean(np.concatenate([seg(t0 + 0.03, t1 - 0.02), seg(t2 + 0.03, min(t3, t2 + 0.12))]) ** 2))
        cs = seg(t1 + 0.005, t2 - 0.005)
        # the release: the biggest rise of the >2 kHz energy within 30 ms of the vowel's start
        w = seg(t2 - 0.03, t2 + 0.02)
        hp = np.diff(w, prepend=w[0])
        e = np.array([np.sum(hp[k:k + int(0.001 * sr)] ** 2) for k in range(0, len(hp) - int(0.001 * sr), int(0.001 * sr))])
        el = 10 * np.log10(e + 1e-12)
        r_at = t2 - 0.03 + 0.001 * (int(np.argmax(np.diff(el))) + 1) if len(el) > 2 else t2
        rel = seg(r_at - 0.001, r_at + 0.011)
        f, X = spectrum(rel, sr)
        band = (f > 500) & (f < 8000)
        pk = f[band][np.argmax(X[band])]
        cen = float((f[band] * X[band] ** 2).sum() / (X[band] ** 2).sum())
        # F2 at voicing onset: the first periodic 20 ms after the release
        von = r_at
        for k in range(30):
            fr = seg(r_at + 0.002 * k, r_at + 0.002 * k + 0.02)
            if len(fr) > 10 and voiced_share(fr, sr) > 0.5:
                von = r_at + 0.002 * k
                break
        on = lpc_formants(seg(von + 0.005, von + 0.03), sr)
        mid = lpc_formants(seg(0.5 * (t2 + t3) - 0.0125, 0.5 * (t2 + t3) + 0.0125), sr)
        f2on = on[1] if len(on) > 1 else float("nan")
        f2mid = mid[1] if len(mid) > 1 else float("nan")
        crms = np.sqrt(np.mean(cs ** 2)) if len(cs) else 0
        nd = 20 * np.log10(crms / vowel_rms + 1e-9)
        fc, Xc = spectrum(cs, sr) if len(cs) > 64 else (f, X)
        bc = (fc > 300) & (fc < 10000)
        ncen = float((fc[bc] * Xc[bc] ** 2).sum() / (Xc[bc] ** 2).sum())
        vs = voiced_share(cs, sr) if len(cs) > int(0.03 * sr) else float("nan")
        hh = h1h2(seg(t2 + 0.005, t2 + 0.035), sr)
        name = f"{v1}{c}{v2}".replace("́", "")
        print(f"{name:8} {pk:8.0f} {cen:6.0f} {f2on:6.0f} {f2mid:6.0f} {nd:6.1f} {ncen:6.0f} {vs*100:5.0f}% {hh:6.1f}")


if __name__ == "__main__":
    main()
