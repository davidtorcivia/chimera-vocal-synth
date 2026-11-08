#!/usr/bin/env python3
"""Restoration v2: repairs to the ElevenLabs conversions of the Chimera renders, inside known phone spans (the
conversions are time-aligned with their sources), always in the target voice.

EQ (`eq_*`):
  eq_voice   per-voice tonal correction toward the long-term spectrum of clean professional voices of the same
             gender (full/eq-reference.json), third-octave, smoothed, boosts only (fills holes, never cuts),
             capped at +MAX_EQ dB; linear phase
  eq_bursts  p t k (and kʷ): 2-8 kHz presence in the 18 ms after the release, +G dB
  eq_s       s spans: cut 2.5-5 kHz, lift 6.5-10 kHz (the female voices' s slides toward ʃ)
Resynthesis (WORLD analysis of a short segment, edited, resynthesized, cross-faded back):
  fix_kw     kʷ gʷ: formants lowered at the vowel onset (rounding), fading out over 50 ms
  fix_trill  r: amplitude dips above 400 Hz at the source's own contact times (synthetic at 25 Hz if none)
  fix_murmur vowels after bʰ dʰ gʰ ɟʰ gʷʰ: breathiness (aperiodicity up, H1 up) over the first third, fading out
"""
import json
from pathlib import Path

import numpy as np
import parselmouth
import pyworld as pw
from scipy.signal import butter, sosfiltfilt, firwin2, fftconvolve

FULL = Path.home() / "AI/chimera-pie-data/full"
MAX_EQ = 4.0
REF = None


def phones(timing):
    d = json.load(open(timing))
    out = []
    for w in d["words"]:
        for p in w["phones"]:
            b = "".join(c for c in p["phone"] if c.isalpha() or c in "ɡɣʁχəɑ").replace("ɡ", "g")
            b = b[:1] if b else b
            out.append({"b": b, "ph": p["phone"], "t0": p["start"], "t1": p["end"], "lab": p["labial"], "br": p["breathy"],
                        "syl": p["kind"] in ("Ns", "Ls", "Rs"), "kind": p["kind"]})
    return out


def is_vowel(p):
    return p["kind"] in ("V", "Vl", "VL") or p["b"] in "aeiouəɑ"


def ramp_window(n_total, sr, a, b, ramp=0.004):
    """0..1 window over [a, b] seconds with raised-cosine ramps."""
    w = np.zeros(n_total)
    ia, ib = max(0, int((a - ramp) * sr)), min(n_total, int((b + ramp) * sr))
    if ib <= ia:
        return w
    t = np.arange(ia, ib) / sr
    up = np.clip((t - (a - ramp)) / ramp, 0, 1)
    dn = np.clip(((b + ramp) - t) / ramp, 0, 1)
    w[ia:ib] = 0.5 - 0.5 * np.cos(np.pi * np.minimum(up, dn))
    return w


def bandpass(x, sr, lo, hi):
    sos = butter(4, [lo, min(hi, sr / 2 * 0.98)], btype="band", fs=sr, output="sos")
    return sosfiltfilt(sos, x)


def highpass(x, sr, f):
    return sosfiltfilt(butter(2, f, btype="high", fs=sr, output="sos"), x)


# ------------------------------------------------------------------ EQ
def ltas_bands(x, sr, edges):
    n = int(0.04 * sr)
    fr = x[: len(x) // n * n].reshape(-1, n)
    rms = np.sqrt((fr ** 2).mean(1))
    sp = fr[rms > np.percentile(rms, 60)]
    S = (np.abs(np.fft.rfft(sp * np.hanning(n), axis=1)) ** 2).mean(0)
    f = np.fft.rfftfreq(n, 1 / sr)
    b = np.array([S[(f >= edges[i]) & (f < edges[i + 1])].mean() for i in range(len(edges) - 1)])
    return 10 * np.log10(b / S[(f >= 1000) & (f < 4000)].mean())


def voice_curve(clips, sr, gender):
    """The EQ curve (dB per third-octave band) for a voice, from several of its clips."""
    global REF
    REF = REF or json.load(open(FULL / "eq-reference.json"))
    edges = np.array(REF["edges"])
    ref = np.array(REF["ref"][gender])
    meas = np.median(np.array([ltas_bands(x, sr, edges) for x in clips]), axis=0)
    g = ref - meas
    g = np.convolve(np.r_[g[0], g, g[-1]], [0.25, 0.5, 0.25], mode="valid")   # smooth across neighbours
    # fill holes only, never cut: the reference voices are thinner than the ones chosen for their body (owner)
    return edges, np.clip(g, 0.0, MAX_EQ)


def eq_voice(x, sr, edges, gains, ntaps=2049):
    centres = np.sqrt(edges[:-1] * edges[1:])
    freqs = np.r_[0, centres, sr / 2]
    gl = 10 ** (np.r_[gains[0], gains, gains[-1]] / 20)
    h = firwin2(ntaps, freqs / (sr / 2), gl)
    y = fftconvolve(x, h, mode="same")
    return y


def eq_bursts(x, sr, ph, gain_db):
    if gain_db <= 0:
        return x
    bp = bandpass(x, sr, 2000, 8000)
    w = np.zeros(len(x))
    for i, p in enumerate(ph):
        nx = ph[i + 1] if i + 1 < len(ph) else None
        if p["b"] in ("p", "t", "k") and nx is not None and abs(nx["t0"] - p["t1"]) < 0.02:
            w = np.maximum(w, ramp_window(len(x), sr, p["t1"], p["t1"] + 0.018, 0.003))
    return x + (10 ** (gain_db / 20) - 1) * bp * w


def eq_s(x, sr, ph, cut_db, lift_db):
    if cut_db == 0 and lift_db == 0:
        return x
    lo, hi = bandpass(x, sr, 2500, 5000), bandpass(x, sr, 6500, 10000)
    w = np.zeros(len(x))
    for p in ph:
        if p["b"] == "s":
            w = np.maximum(w, ramp_window(len(x), sr, p["t0"] + 0.005, p["t1"] - 0.005, 0.006))
    return x + ((10 ** (-cut_db / 20) - 1) * lo + (10 ** (lift_db / 20) - 1) * hi) * w


# ------------------------------------------------------------------ resynthesis
def _world_segment(x, sr, a, b, edit, pad=0.03):
    """Analyse [a-pad, b+pad] with WORLD, apply edit(times, f0, sp, ap) in place, resynthesize, cross-fade back."""
    ia, ib = max(0, int((a - pad) * sr)), min(len(x), int((b + pad) * sr))
    seg = x[ia:ib].astype(np.float64)
    if len(seg) < int(0.04 * sr):
        return x
    fp = 2.0
    f0, t = pw.harvest(seg, sr, f0_floor=55, f0_ceil=500, frame_period=fp)
    sp = pw.cheaptrick(seg, f0, t, sr)
    ap = pw.d4c(seg, f0, t, sr)
    times = t + ia / sr
    if not edit(times, f0, sp, ap):
        return x
    y = pw.synthesize(f0, sp, ap, sr, frame_period=fp)[: len(seg)]
    y = np.pad(y, (0, len(seg) - len(y)))
    # match level, then cross-fade 10 ms at both ends of [a, b]
    w = ramp_window(len(seg), sr, a - ia / sr, b - ia / sr, 0.01)
    out = x.copy()
    out[ia:ib] = seg * (1 - w) + y * w
    return out


def fix_kw(x, sr, ph, stretch=2.0, head=0.025):
    """kʷ gʷ: the converter shortens the rounding transition; slow the vowel's first `head` s by `stretch`, taking
    the time back from the rest of the vowel (its length is kept). Only the voice's own frames are used."""
    for i, p in enumerate(ph):
        nx = ph[i + 1] if i + 1 < len(ph) else None
        if p["b"] in ("k", "g") and p["lab"] and nx is not None and is_vowel(nx) and abs(nx["t0"] - p["t1"]) < 0.02:
            on, off = nx["t0"], nx["t1"]
            if off - on < 0.07:
                continue

            def edit(times, f0, sp, ap, on=on, off=off):
                sel = np.where((times >= on) & (times < off))[0]
                if len(sel) < 10:
                    return False
                tv = times[sel] - on
                L, H = off - on, head * stretch
                # output time u -> input time: [0, H] -> [0, head], [H, L] -> [head, L]
                src_t = np.where(tv < H, tv / stretch, head + (tv - H) * (L - head) / (L - H))
                k = np.interp(src_t + on, times, np.arange(len(times)))
                k0 = np.clip(np.floor(k).astype(int), 0, len(times) - 2)
                fr = (k - k0)[:, None]
                lsp = np.log(sp + 1e-20)
                f0c, spc, apc = f0.copy(), lsp.copy(), ap.copy()
                f0[sel] = (1 - fr[:, 0]) * f0c[k0] + fr[:, 0] * f0c[k0 + 1]
                sp[sel] = np.exp((1 - fr) * spc[k0] + fr * spc[k0 + 1])
                ap[sel] = (1 - fr) * apc[k0] + fr * apc[k0 + 1]
                return True

            x = _world_segment(x, sr, on - 0.005, off + 0.005, edit)
    return x


def source_contacts(src, sr, a, b):
    snd = parselmouth.Sound(src[int(max(0, a - 0.02) * sr):int((b + 0.02) * sr)], sampling_frequency=sr)
    inten = snd.to_intensity(minimum_pitch=400, time_step=0.001)
    ts = np.arange(0.02, 0.02 + (b - a), 0.001)
    v = np.array([inten.get_value(t) or np.nan for t in ts])
    dips = []
    k = 1
    while k < len(v) - 1:
        if np.isfinite(v[k]) and v[k] < v[k - 1] and v[k] <= v[k + 1]:
            left, right = np.nanmax(v[max(0, k - 15):k]), np.nanmax(v[k + 1:k + 16])
            if min(left, right) - v[k] >= 6:
                dips.append(a + ts[k] - 0.02)
                k += 8
        k += 1
    return dips


def fix_trill(x, sr, ph, src, depth_db=10.0, hf_extra_db=6.0, width=0.010):
    hp = highpass(x, sr, 400)
    g = np.ones(len(x))
    for i, p in enumerate(ph):
        if p["b"] != "r" or p["syl"]:
            continue
        pv = ph[i - 1] if i > 0 else None
        nx = ph[i + 1] if i + 1 < len(ph) else None
        both = pv is not None and nx is not None and is_vowel(pv) and is_vowel(nx)
        mid = 0.5 * (p["t0"] + p["t1"])
        a, b = (mid - 0.055, mid + 0.055) if both else (p["t0"] - 0.01, p["t1"] + 0.01)
        dips = source_contacts(src, sr, a, b)
        if len(dips) < 2 and both:
            dips = [mid - 0.02, mid + 0.02]
        for t in dips:
            n0, n1 = int((t - width) * sr), int((t + width) * sr)
            if n0 < 0 or n1 > len(x):
                continue
            shape = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n1 - n0) / (n1 - n0))   # 0 → 1 → 0
            g[n0:n1] = np.minimum(g[n0:n1], 1 - (1 - 10 ** (-depth_db / 20)) * shape)
    ghf = g ** (1 + hf_extra_db / depth_db)                     # the band above 400 Hz dips further
    return (x - hp) * g + hp * ghf


def fix_murmur(x, sr, ph, ap_mix=0.3, h1_db=6.0, tilt_db=4.0):
    """Breathy release after bʰ dʰ gʰ ɟʰ gʷʰ: over the vowel's first third (fading out), a strong first harmonic, a
    steeper fall above it (tilt) and some breath noise (aperiodicity) below 4 kHz."""
    for i, p in enumerate(ph):
        nx = ph[i + 1] if i + 1 < len(ph) else None
        if p["br"] and p["b"] in ("b", "d", "g", "ɟ") and nx is not None and is_vowel(nx) and abs(nx["t0"] - p["t1"]) < 0.02:
            on, dur = nx["t0"], max(0.05, (nx["t1"] - nx["t0"]) / 3)

            def edit(times, f0, sp, ap, on=on, dur=dur):
                nb = sp.shape[1]
                fr = np.arange(nb) / (nb - 1) * sr / 2
                changed = False
                for k, tt in enumerate(times):
                    tau = tt - on
                    if 0 <= tau <= dur and f0[k] > 0:
                        s_ = 1 - tau / dur
                        low = fr < 4000
                        ap[k, low] = ap[k, low] + (1 - ap[k, low]) * ap_mix * s_
                        g = h1_db * np.exp(-0.5 * ((fr - f0[k]) / (0.4 * f0[k])) ** 2) \
                            - tilt_db * np.clip((fr - 1.5 * f0[k]) / 1500.0, 0, 1)
                        sp[k] *= 10 ** (s_ * g / 10)
                        changed = True
                return changed

            x = _world_segment(x, sr, on, on + dur, edit)
    return x


# ------------------------------------------------------------------ contrast transfer (no resynthesis)
def _stft(x, n=512, hop=128):
    w = np.hanning(n)
    pad = np.pad(x, (n, n))
    frames = (len(pad) - n) // hop + 1
    idx = np.arange(frames)[:, None] * hop + np.arange(n)[None, :]
    return np.fft.rfft(pad[idx] * w, axis=1), w


def _istft(X, length, n=512, hop=128, w=None):
    w = np.hanning(n) if w is None else w
    fr = np.fft.irfft(X, n=n, axis=1) * w
    out = np.zeros((X.shape[0] - 1) * hop + n)
    norm = np.zeros_like(out)
    for i, f in enumerate(fr):
        out[i * hop:i * hop + n] += f
        norm[i * hop:i * hop + n] += w ** 2
    out = out / np.maximum(norm, 1e-8)
    return out[n:n + length]


def _bands(sr, n, lo=80, hi=None, per_oct=3):
    hi = hi or sr / 2 * 0.95
    edges = [lo]
    while edges[-1] * 2 ** (1 / per_oct) < hi:
        edges.append(edges[-1] * 2 ** (1 / per_oct))
    f = np.fft.rfftfreq(n, 1 / sr)
    return np.array(edges), f


def contrast_transfer(x, src, sr, spans, n=512, hop=128, max_db=9.0):
    """For each span (a, b, ref_a, ref_b, alpha): give the converted signal, inside [a, b], the source's spectral
    contrast against its own reference stretch [ref_a, ref_b] (third-octave bands), raised to `alpha`, as a gain on the
    STFT magnitude with the original phase. Gains are smoothed over time and capped at ±max_db."""
    X, w = _stft(x, n, hop)
    S, _ = _stft(src, n, hop)
    edges, f = _bands(sr, n)
    G = np.zeros(X.shape)                                     # dB
    W = np.zeros(X.shape[0])
    tfr = (np.arange(X.shape[0]) * hop - n) / sr + n / 2 / sr   # frame centre times
    bi = [np.where((f >= edges[i]) & (f < edges[i + 1]))[0] for i in range(len(edges) - 1)]
    cen = np.sqrt(edges[:-1] * edges[1:])

    def bandpow(M, rows):
        P = np.abs(M[rows]) ** 2
        return np.array([P[:, ix].mean() if len(ix) else np.nan for ix in bi]) if P.ndim == 2 else None

    for (a, b, ra, rb, alpha) in spans:
        rows = np.where((tfr >= a) & (tfr <= b))[0]
        rref = np.where((tfr >= ra) & (tfr <= rb))[0]
        if len(rows) == 0 or len(rref) < 2:
            continue
        refS = np.array([(np.abs(S[rref][:, ix]) ** 2).mean() for ix in bi]) + 1e-12
        refX = np.array([(np.abs(X[rref][:, ix]) ** 2).mean() for ix in bi]) + 1e-12
        for r in rows:
            cs = np.array([(np.abs(S[r, ix]) ** 2).mean() for ix in bi]) / refS
            cx = np.array([(np.abs(X[r, ix]) ** 2).mean() for ix in bi]) / refX
            gdb = alpha * 10 * np.log10((cs + 1e-12) / (cx + 1e-12))
            gdb = np.clip(np.nan_to_num(gdb), -max_db, max_db)
            G[r] = np.interp(np.log(np.maximum(f, 1)), np.log(cen), gdb)
            W[r] = 1.0
    # smooth over time (5 frames) so the gain has no steps
    k = np.hanning(7); k /= k.sum()
    Gs = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, G * W[:, None])
    Ws = np.convolve(W, k, mode="same")
    Gs = Gs / np.maximum(Ws, 1e-6)[:, None] * np.minimum(Ws, 1)[:, None]
    Y = X * 10 ** (Gs / 20)
    return _istft(Y, len(x), n, hop, w)


def burst_spans(ph, dur=0.02, alpha=1.0):
    """The 20 ms after each p t k (plain or labial) release, against the middle of the following vowel."""
    out = []
    for i, p in enumerate(ph):
        nx = ph[i + 1] if i + 1 < len(ph) else None
        if p["b"] in ("p", "t", "k") and nx is not None and is_vowel(nx) and abs(nx["t0"] - p["t1"]) < 0.02:
            vm0, vm1 = nx["t0"] + 0.3 * (nx["t1"] - nx["t0"]), nx["t1"] - 0.3 * (nx["t1"] - nx["t0"])
            out.append((p["t1"] - 0.004, p["t1"] + dur, vm0, vm1, alpha))
    return out


# ------------------------------------------------------------------ trill by envelope transfer
def trill_windows(ph):
    out = []
    for i, p in enumerate(ph):
        if p["b"] != "r" or p["syl"]:
            continue
        pv = ph[i - 1] if i > 0 else None
        nx = ph[i + 1] if i + 1 < len(ph) else None
        both = pv is not None and nx is not None and is_vowel(pv) and is_vowel(nx)
        mid = 0.5 * (p["t0"] + p["t1"])
        out.append((mid - 0.055, mid + 0.055) if both else (p["t0"] - 0.01, p["t1"] + 0.01))
    return out


def fix_trill_env(x, guide, sr, ph, lo=500.0, hi=3500.0, floor_db=-20.0, ceil_db=3.0, smooth=0.0025):
    """Give the converted r, above `lo`, the closure rhythm of `guide` (the Chimera source, or a sibling conversion
    of the same sentence): the guide's lo-hi band envelope normalized to its peak in the window, over the converted
    one's. The band below `lo` (voicing) is untouched. Zero-phase band split; gains ramped."""
    from scipy.signal import hilbert
    low = sosfiltfilt(butter(4, lo, btype="low", fs=sr, output="sos"), x)
    rest = x - low
    gm = bandpass(guide, sr, lo, hi)
    xm = bandpass(x, sr, lo, hi)
    k = max(1, int(smooth * sr))
    ker = np.hanning(2 * k + 1); ker /= ker.sum()
    env = lambda y: np.convolve(np.abs(hilbert(y)), ker, mode="same")
    eg, ex = env(gm), env(xm)
    gain_db = np.zeros(len(x))
    wsum = np.zeros(len(x))
    for a, b in trill_windows(ph):
        ia, ib = max(0, int(a * sr)), min(len(x), int(b * sr))
        if ib - ia < int(0.02 * sr):
            continue
        ng = eg[ia:ib] / (eg[ia:ib].max() + 1e-12)
        nx_ = ex[ia:ib] / (ex[ia:ib].max() + 1e-12)
        gd = np.clip(20 * np.log10((ng + 1e-6) / (nx_ + 1e-6)), floor_db, ceil_db)
        w = ramp_window(len(x), sr, a, b, 0.008)[ia:ib]
        gain_db[ia:ib] += gd * w
        wsum[ia:ib] = np.maximum(wsum[ia:ib], w)
    return low + rest * 10 ** (gain_db / 20)


# ------------------------------------------------------------------ formant edits in the LPC domain (no vocoder)
def _lpc(frame, order):
    """Autocorrelation LPC (Levinson-Durbin); returns a with a[0] = 1."""
    r = np.correlate(frame, frame, mode="full")[len(frame) - 1:len(frame) + order]
    if r[0] <= 0:
        return np.r_[1.0, np.zeros(order)]
    r = r.copy(); r[0] *= 1.0001
    a = np.zeros(order + 1); a[0] = 1.0; e = r[0]
    for i in range(1, order + 1):
        k = -(r[i] + np.dot(a[1:i], r[i - 1:0:-1])) / e
        a[1:i + 1] = a[1:i + 1] + k * np.r_[a[i - 1:0:-1], 1.0]
        e *= (1 - k * k)
        if e <= 0:
            break
    return a


def _shift_formants(a, fs, scale, which=(1, 2)):
    """Scale the frequencies of the formant-like poles (bandwidth < 600 Hz, 200-4000 Hz) with index in `which`
    (0 = F1) by `scale`; returns the new polynomial."""
    roots = np.roots(a)
    pos = [i for i, z in enumerate(roots) if np.imag(z) > 0]
    cand = []
    for i in pos:
        z = roots[i]
        f = np.angle(z) * fs / (2 * np.pi)
        bw = -np.log(max(abs(z), 1e-9)) * fs / np.pi
        if 200 < f < 4000 and bw < 600:
            cand.append((f, i))
    cand.sort()
    new = roots.copy()
    for rank, (f, i) in enumerate(cand):
        if rank in which:
            z = roots[i]
            ang = np.angle(z) * scale
            zn = abs(z) * np.exp(1j * ang)
            new[i] = zn
            # its conjugate
            j = int(np.argmin(np.abs(roots - np.conj(z))))
            new[j] = np.conj(zn)
    return np.real(np.poly(new))


def lpc_formant_edit(x, sr, a_t, b_t, scale_fn, order, which=(1, 2), fs2=11025, hop=0.005, win=0.02):
    """Edit formants of x between a_t and b_t: low band (< 5 kHz, at fs2) inverse-filtered block by block with its
    own LPC, re-filtered with poles moved by scale_fn(t) (1 = unchanged, identity), high band untouched."""
    from scipy.signal import resample_poly, lfilter
    from math import gcd
    pad = 0.04
    ia, ib = max(0, int((a_t - pad) * sr)), min(len(x), int((b_t + pad) * sr))
    seg = x[ia:ib]
    lowp = butter(8, 4800, btype="low", fs=sr, output="sos")
    low = sosfiltfilt(lowp, seg)
    high = seg - low
    g = gcd(sr, fs2)
    lo2 = resample_poly(low, fs2 // g, sr // g)
    H, W = int(hop * fs2), int(win * fs2)
    out = np.zeros_like(lo2)
    za = np.zeros(order); zs = np.zeros(order)
    w = np.hanning(W)
    for k0 in range(0, len(lo2), H):
        blk = lo2[k0:k0 + H]
        c = k0 + H // 2
        fr = lo2[max(0, c - W // 2):max(0, c - W // 2) + W]
        if len(fr) < W:
            fr = np.pad(fr, (0, W - len(fr)))
        a = _lpc(fr * w, order)
        t = (ia / sr) + (c / fs2)
        sc = scale_fn(t)
        a2 = a if abs(sc - 1) < 1e-4 else _shift_formants(a, fs2, sc, which)
        e, za = lfilter(a, [1.0], blk, zi=za)
        y, zs = lfilter([1.0], a2, e, zi=zs)
        out[k0:k0 + len(blk)] = y
    back = resample_poly(out, sr // g, fs2 // g)[: len(seg)]
    back = np.pad(back, (0, len(seg) - len(back)))
    y = x.copy()
    edited = back + high
    wgt = ramp_window(len(seg), sr, a_t - ia / sr, b_t - ia / sr, 0.01)
    y[ia:ib] = seg * (1 - wgt) + edited * wgt
    return y


def fix_kw_lpc(x, sr, ph, depth=0.15, dur=0.06, male=True):
    """kʷ gʷ: lower F2 and F3 at the following vowel's onset by `depth` (fraction), fading to 0 over `dur`."""
    order = 12 if male else 10
    for i, p in enumerate(ph):
        nx = ph[i + 1] if i + 1 < len(ph) else None
        if p["b"] in ("k", "g") and p["lab"] and nx is not None and is_vowel(nx) and abs(nx["t0"] - p["t1"]) < 0.02:
            on = nx["t0"]
            fn = lambda t, on=on: 1 - depth * float(np.clip(1 - (t - on) / dur, 0, 1)) if t >= on - 0.01 else 1.0
            x = lpc_formant_edit(x, sr, on - 0.01, on + dur, fn, order)
    return x


def fix_kw_own_w(x, sr, ph, dur=0.03, strength=1.0, max_db=12.0, n=1024, hop=128):
    """kʷ gʷ: over the first `dur` s of the following vowel (fading out), a spectral gain that turns the vowel's own
    spectrum toward the same clip's w/u spectrum (this voice's own rounding), smoothed in third-octave bands.
    STFT magnitude only; original phase."""
    X, w = _stft(x, n, hop)
    edges, f = _bands(sr, n)
    bi = [np.where((f >= edges[i]) & (f < edges[i + 1]))[0] for i in range(len(edges) - 1)]
    cen = np.sqrt(edges[:-1] * edges[1:])
    tfr = (np.arange(X.shape[0]) * hop - n) / sr + n / 2 / sr
    def bandspec(rows):
        P = np.abs(X[rows]) ** 2
        return np.array([P[:, ix].mean() if len(ix) else 1e-12 for ix in bi]) + 1e-12
    # the voice's own rounded reference: middles of w and u in this clip
    rw = [r for p in ph if p["b"] in ("w", "u") for r in np.where((tfr >= p["t0"] + 0.3 * (p["t1"] - p["t0"])) & (tfr <= p["t1"] - 0.3 * (p["t1"] - p["t0"])))[0]]
    if len(rw) < 3:
        return x
    Ew = bandspec(np.array(rw))
    G = np.zeros(X.shape)
    for i, p in enumerate(ph):
        nx = ph[i + 1] if i + 1 < len(ph) else None
        if p["b"] in ("k", "g") and p["lab"] and nx is not None and is_vowel(nx) and abs(nx["t0"] - p["t1"]) < 0.02:
            on, off = nx["t0"], nx["t1"]
            mid = np.where((tfr >= on + 0.3 * (off - on)) & (tfr <= off - 0.2 * (off - on)))[0]
            if len(mid) < 2:
                continue
            Ev = bandspec(mid)
            # shape only (level-neutral): normalise both to their 300-3000 Hz mean
            m = (cen > 300) & (cen < 3000)
            shape = 10 * np.log10(Ew / Ew[m].mean()) - 10 * np.log10(Ev / Ev[m].mean())
            shape = np.clip(shape, -max_db, max_db)
            gbin = np.interp(np.log(np.maximum(f, 1)), np.log(cen), shape)
            for r in np.where((tfr >= on - 0.005) & (tfr <= on + dur))[0]:
                a = strength * float(np.clip(1 - (tfr[r] - on) / dur, 0, 1))
                G[r] = np.maximum(G[r], 0) if False else a * gbin
    k = np.hanning(5); k /= k.sum()
    G = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, G)
    return _istft(X * 10 ** (G / 20), len(x), n, hop, w)


def fix_murmur_noise(x, sr, ph, noise_db=-15.0, h1_db=3.0, tilt_db=3.0, seed=0, n=1024, hop=128):
    """Breathy release after bʰ dʰ gʰ ɟʰ gʷʰ, no vocoder: over the vowel's first third (fading out), (1) aspiration
    noise high-passed at 1.5 kHz, shaped by the vowel's own LPC envelope and gated pitch-synchronously (peak in the
    open phase, Hermes 1991), mixed `noise_db` re the vowel's own >2 kHz level; (2) a small H1 lift and high-shelf cut
    as an STFT gain (Klatt & Klatt 1990: noise is the main cue; an H1 boost alone reads as nasal in a high voice)."""
    from scipy.signal import lfilter
    rng = np.random.default_rng(seed)
    y = x.copy()
    spans = []
    for i, p in enumerate(ph):
        nx = ph[i + 1] if i + 1 < len(ph) else None
        if p["br"] and p["b"] in ("b", "d", "g", "ɟ") and nx is not None and is_vowel(nx) and abs(nx["t0"] - p["t1"]) < 0.02:
            spans.append((nx["t0"], nx["t0"] + max(0.05, (nx["t1"] - nx["t0"]) / 3)))
    if not spans:
        return x
    snd = parselmouth.Sound(x, sampling_frequency=sr)
    pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
    npts = parselmouth.praat.call(pp, "Get number of points")
    pulses = np.array([parselmouth.praat.call(pp, "Get time from index", k + 1) for k in range(npts)])
    f0_at = lambda t: (1.0 / np.median(np.diff(pulses[(pulses > t - 0.03) & (pulses < t + 0.03)]))) if np.sum((pulses > t - 0.03) & (pulses < t + 0.03)) > 2 else 200.0
    hp = butter(4, 1500, btype="high", fs=sr, output="sos")
    for a, b in spans:
        ia, ib = int(a * sr), min(len(x), int(b * sr))
        if ib - ia < int(0.03 * sr):
            continue
        seg = x[ia:ib]
        aa = _lpc(seg * np.hanning(len(seg)), 30)
        nz = lfilter([1.0], aa, sosfiltfilt(hp, rng.standard_normal(len(seg))))
        # pitch-synchronous gate: raised cosine per period, peaking 60 % into the period
        t = (ia + np.arange(len(seg))) / sr
        gate = np.full(len(seg), 0.5)
        pk = pulses[(pulses >= a - 0.02) & (pulses <= b + 0.02)]
        for p0, p1 in zip(pk[:-1], pk[1:]):
            m = (t >= p0) & (t < p1)
            ph_ = (t[m] - p0) / (p1 - p0)
            gate[m] = 0.5 + 0.5 * np.cos(2 * np.pi * (ph_ - 0.6))
        nz *= gate
        ref = np.sqrt(np.mean(sosfiltfilt(butter(4, 2000, btype="high", fs=sr, output="sos"), seg) ** 2)) + 1e-9
        nz *= ref * 10 ** (noise_db / 20) / (np.sqrt(np.mean(nz ** 2)) + 1e-12)
        fade = np.clip(1 - (t - a) / (b - a), 0, 1)
        fade *= np.clip((t - a) / 0.005, 0, 1)
        y[ia:ib] += nz * fade
    # H1 lift and tilt as an STFT gain over the same spans
    X, w = _stft(y, n, hop)
    f = np.fft.rfftfreq(n, 1 / sr)
    tfr = (np.arange(X.shape[0]) * hop - n) / sr + n / 2 / sr
    G = np.zeros(X.shape)
    for a, b in spans:
        f0 = f0_at(0.5 * (a + b))
        g = h1_db * np.exp(-0.5 * ((f - f0) / (0.5 * f0)) ** 2) - tilt_db * np.clip((f - 2000) / 1000, 0, 1)
        for r in np.where((tfr >= a) & (tfr <= b))[0]:
            G[r] = g * float(np.clip(1 - (tfr[r] - a) / (b - a), 0, 1))
    k = np.hanning(5); k /= k.sum()
    G = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, G)
    return _istft(X * 10 ** (G / 20), len(y), n, hop, w)
