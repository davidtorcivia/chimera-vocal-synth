"""The mel mapper: Chimera log-mel (128 x T) and a voice id in, the voice's log-mel out, frame for frame.
A stack of residual blocks (layer norm, depthwise conv over time, GELU, pointwise; then a feed-forward), so the
receptive field is set by kernel and depth: ±(kernel-1)/2 x layers frames (11.6 ms each). The output is the source
plus a predicted correction, so an untrained mapper passes Chimera through."""
import torch
from torch import nn

MEAN, STD = -5.0, 2.5            # log-mel scaling (BigVGAN's log magnitudes run about -11.5 .. 2)


class Block(nn.Module):
    def __init__(self, d, k, drop):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d); self.pw = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d); self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x):                         # x: B, T, d
        h = self.n1(x).transpose(1, 2)
        h = self.dw(h).transpose(1, 2)
        x = x + self.drop(self.pw(nn.functional.gelu(h)))
        return x + self.drop(self.ff(self.n2(x)))


class Mapper(nn.Module):
    def __init__(self, n_mel=128, d=384, layers=8, kernel=7, n_spk=4, drop=0.1):
        super().__init__()
        self.inp = nn.Linear(n_mel, d); self.spk = nn.Embedding(n_spk, d)
        self.blocks = nn.ModuleList([Block(d, kernel, drop) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_mel))
        nn.init.zeros_(self.out[1].weight); nn.init.zeros_(self.out[1].bias)
        self.cfg = dict(n_mel=n_mel, d=d, layers=layers, kernel=kernel, n_spk=n_spk, drop=drop)

    def forward(self, mel, spk):                  # mel: B, 128, T (log); spk: B
        x = ((mel - MEAN) / STD).transpose(1, 2)
        h = self.inp(x) + self.spk(spk)[:, None]
        for b in self.blocks:
            h = b(h)
        return mel + (self.out(h) * STD).transpose(1, 2)

    def frames_each_side(self):
        return (self.cfg["kernel"] - 1) // 2 * self.cfg["layers"]
