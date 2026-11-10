"""v2 of the Chimera -> natural converter: a conditional flow-matching model of the target voice's log-mel, given the
Chimera render's log-mel (frame for frame: the pairs are sample-aligned) and the voice. Its mels go to the released
BigVGAN-v2 vocoder, frozen: in the owner's listening test (2026-11-09) BigVGAN fine-tuned with v1's mapper was much
bumpier than the released one even on the paid audio's own mels, while the released one on real mels was next to the
reference. So the vocoder stays as released and all the learning is here, where a flow model gives sharp mels instead of
the L1 mapper's smoothed ones (v0) or a mapper co-adapted to a degraded vocoder (v1).

The network: per frame, the noisy target mel x_t and the source mel, both scaled, projected to d; DiT blocks (adaLN-zero
modulation by the flow time and the voice; self-attention with rotary positions; a depthwise conv and an MLP); a
zero-initialised projection to the velocity. Rectified flow: x_t = (1 - (1 - s) t) x0 + t x1, target x1 - (1 - s) x0.
Classifier-free guidance is trained by dropping the source and the voice together (p_uncond); sampling is Euler.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

MEAN, STD = -5.0, 2.5            # log-mel scaling, as pie/sts/model.py (BigVGAN's log magnitudes run about -11.5 .. 2)
SIGMA_MIN = 1e-4


def rope(x, base=10000.0):       # x: B, H, T, D (D even): rotary positions over T
    T, D = x.shape[2], x.shape[3]
    f = base ** (-torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D)
    a = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * f[None]
    c, s = a.cos()[None, None].to(x.dtype), a.sin()[None, None].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, heads, k):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(d, elementwise_affine=False); self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False); self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))
        self.mod = nn.Linear(d, 6 * d); nn.init.zeros_(self.mod.weight); nn.init.zeros_(self.mod.bias)

    def forward(self, x, c, pad):                 # x: B, T, d; c: B, d; pad: B, T (True = padding) or None
        s1, b1, g1, s2, b2, g2 = self.mod(F.silu(c))[:, None].chunk(6, -1)
        h = self.n1(x) * (1 + s1) + b1
        B, T, d = h.shape
        q, k, v = self.qkv(h).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q), rope(k)
        mask = None if pad is None else (~pad)[:, None, None, :]
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, T, d)
        x = x + g1 * self.o(h)
        h = self.n2(x) * (1 + s2) + b2
        h = self.dw(h.transpose(1, 2)).transpose(1, 2)
        return x + g2 * self.ff(h)


class FlowNet(nn.Module):
    def __init__(self, n_mel=128, d=512, layers=12, heads=8, kernel=5, n_spk=4):
        super().__init__()
        self.cfg = dict(n_mel=n_mel, d=d, layers=layers, heads=heads, kernel=kernel, n_spk=n_spk)
        self.inp = nn.Linear(2 * n_mel, d)
        self.spk = nn.Embedding(n_spk + 1, d)                  # the last id: no voice (classifier-free guidance)
        self.t = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, heads, kernel) for _ in range(layers)])
        self.nf = nn.LayerNorm(d, elementwise_affine=False); self.mf = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, n_mel)
        for m in (self.mf, self.out):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def temb(self, t):                            # t: B in [0, 1]
        f = torch.exp(-math.log(10000) * torch.arange(128, device=t.device, dtype=torch.float32) / 128)
        a = 1000 * t[:, None].float() * f[None]
        return self.t(torch.cat([a.sin(), a.cos()], -1))

    def forward(self, xt, src, spk, t, pad=None):
        """xt, src: B, 128, T (scaled); spk: B (n_spk = none); t: B. Returns the velocity, B, 128, T."""
        h = self.inp(torch.cat([xt, src], 1).transpose(1, 2))
        c = self.temb(t) + self.spk(spk)
        for b in self.blocks:
            h = b(h, c, pad)
        s, b_ = self.mf(F.silu(c))[:, None].chunk(2, -1)
        return self.out(self.nf(h) * (1 + s) + b_).transpose(1, 2)


def scale(mel):
    return (mel - MEAN) / STD


def unscale(x):
    return x * STD + MEAN


@torch.no_grad()
def sample(net, src_mel, spk, steps=32, cfg=0.0, temp=1.0, generator=None):
    """src_mel: B, 128, T (log-mel); spk: B. Returns the generated log-mel, B, 128, T (Euler, steps steps)."""
    src = scale(src_mel)
    x = torch.randn(src.shape, device=src.device, generator=generator) * temp
    null_spk = torch.full_like(spk, net.cfg["n_spk"])
    ts = torch.linspace(0, 1, steps + 1, device=src.device)
    for i in range(steps):
        t = ts[i].expand(src.shape[0])
        v = net(x, src, spk, t)
        if cfg:
            v = v + cfg * (v - net(x, torch.zeros_like(src), null_spk, t))
        x = x + (ts[i + 1] - ts[i]) * v
    return unscale(x)
