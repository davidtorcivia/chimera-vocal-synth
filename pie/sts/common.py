"""Shared pieces of the Chimera -> natural speech-to-speech converter (pie/sts/, Phase 4; runs in WSL on tr-desktop).

Data: each Chimera corpus v2 render (JD3 body for the male voices, W02 body for the female ones) paired with its
restored conversion (restoration v2) in each voice that converted it. Timing survives conversion sample for sample
(lag 0), so source and target frames line up one to one. Mels are BigVGAN-v2 44.1 kHz's own: 128 bands, n_fft 2048,
hop 512, window 2048, fmin 0, fmax Nyquist, log of the magnitude (clamped at 1e-5)."""
import sys
from pathlib import Path

ROOT = Path("/root/sts")
BIGVGAN = ROOT / "bigvgan"
RENDERS = ROOT / "renders"                                   # chimera-v2/renders (44.1 kHz, peak 0.5)
RESTORED = Path("/root/bwe-work2/restored-v2")               # restored-v2/<TAG>/<clip>__<SPK>.wav (44.1 kHz)
LISTS = Path("/mnt/q/AI/PIE/train/kokoro/training")          # v2_{train,val}_list.txt: clip__TAG.wav (gate-passed)
DATA = ROOT / "data"
TAGS = ["ZAZY", "VALERIA", "ANTONIO", "W02"]
SPK = {"ZAZY": "JD3-PIE", "ANTONIO": "JD3-PIE", "VALERIA": "W02-PIE", "W02": "W02-PIE"}
SR, HOP, NMEL = 44100, 512, 128

sys.path.insert(0, str(BIGVGAN))


def mel(y, device="cpu"):
    """BigVGAN's log-mel of a float waveform (numpy or tensor), shape (128, frames)"""
    import torch
    from meldataset import mel_spectrogram
    t = torch.as_tensor(y, dtype=torch.float32, device=device)[None]
    return mel_spectrogram(t, 2048, NMEL, SR, HOP, 2048, 0, None, center=False)[0]


def vocoder(device="cuda", weight_norm=False):
    """BigVGAN-v2 44.1 kHz from its local files (its from_pretrained wrapper does not fit this huggingface_hub)"""
    import json, torch, bigvgan
    from env import AttrDict
    m = bigvgan.BigVGAN(AttrDict(json.load(open(BIGVGAN / "config.json"))), use_cuda_kernel=False)
    m.load_state_dict(torch.load(BIGVGAN / "bigvgan_generator.pt", map_location="cpu", weights_only=True)["generator"])
    if not weight_norm:
        m.remove_weight_norm()
    return m.eval().to(device)


def pairs(split):
    """[(tag, stem)] for a split: stem = clip__SPK"""
    out = []
    for l in open(LISTS / f"v2_{split}_list.txt", encoding="utf-8"):
        if l.strip():
            clip, tag = l.split("|")[0][:-4].rsplit("__", 1)
            out.append((tag, f"{clip}__{SPK[tag]}"))
    return out
