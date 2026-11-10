"""v4: the mapper (pie/sts/model.py, with self-attention layers for whole-line context) trained in the mel domain alone:
masked L1 to the target mel plus an adversarial loss from a patch discriminator on log-mels, so its mels are sharp rather
than averaged, and deterministic, so the harmonics follow the source (no v2 doubling). No vocoder in the loop: the released
BigVGAN-v2 renders the mels at conversion (pie/sts/convert.py, the mapper branch), as it renders the paid audio's own mels
almost indistinguishably from the paid audio (owner's ranking, 2026-11-09).

The pairs in CPU memory (--data_gpu: on the GPU, as train_flow.py); the accuracy mask (pie/sts/mask.py) zeroes the L1 on the masked frames, the
adversarial loss sees every frame. Stage A (--l1_steps): L1 only; then L1 x --lambda_l1 + adversarial.
Checkpoints OUT/m_<step>.pt (EMA mapper, {"model", "cfg"}: convert.py reads them like an L1 mapper); OUT/last.pt; OUT/STOP.
    python pie/sts/train_mgan.py OUT [--steps 60000] [--batch 32] [--crop 256] [--mask mask.json]
"""
import argparse, glob, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from torch import nn
import torch.nn.functional as F
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C
from model import Mapper

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--steps", type=int, default=60000); ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--crop", type=int, default=256); ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--lr_d", type=float, default=2e-4)
ap.add_argument("--d", type=int, default=512); ap.add_argument("--layers", type=int, default=12); ap.add_argument("--kernel", type=int, default=7)
ap.add_argument("--attn_every", type=int, default=3); ap.add_argument("--l1_steps", type=int, default=4000); ap.add_argument("--lambda_l1", type=float, default=20.0)
ap.add_argument("--lambda_fm", type=float, default=2.0); ap.add_argument("--ema", type=float, default=0.999)
ap.add_argument("--mask", default=""); ap.add_argument("--mem", type=float, default=0.1)
ap.add_argument("--data_gpu", action="store_true", help="hold the pairs on the GPU (faster; ~2.5 GB)")
ap.add_argument("--save_every", type=int, default=5000); ap.add_argument("--val_every", type=int, default=1000)
a = ap.parse_args()
dev = "cuda"
torch.cuda.set_per_process_memory_fraction(a.mem, 0)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
SPK = {t: i for i, t in enumerate(C.TAGS)}; SIL = -11.5
MASKS = json.load(open(a.mask)) if a.mask else {}


def frame_weight(key, n):
    w = np.ones(n, np.float16); c = np.arange(n) * C.HOP + 1024
    for s0, s1 in MASKS.get(key, []):
        w[(c >= s0) & (c < s1)] = 0
    return w


def load(split):
    S, T, W, spk, starts, lens, cur = [], [], [], [], [], [], 0
    for f in sorted(glob.glob(str(C.DATA / split / "*.npz"))):
        z = np.load(f); tag, stem = Path(f).stem.split("__", 1); n = z["src"].shape[1]; m = max(n, a.crop)
        S.append(np.pad(z["src"], ((0, 0), (0, m - n)), constant_values=SIL)); T.append(np.pad(z["tgt"], ((0, 0), (0, m - n)), constant_values=SIL))
        W.append(np.pad(frame_weight(f"{tag}/{stem}", n), (0, m - n), constant_values=1)); spk.append(SPK[tag]); starts.append(cur); lens.append(m); cur += m
    g = lambda xs, ax: torch.from_numpy(np.concatenate(xs, ax))            # in CPU memory (--data_gpu: on the GPU)
    D = dict(src=g(S, 1), tgt=g(T, 1), w=g(W, 0), start=torch.tensor(starts), len=torch.tensor(lens), spk=torch.tensor(spk), n=len(lens))
    if a.data_gpu:
        D = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in D.items()}
    return D


train, val = load("train"), load("val"); DD = dev if a.data_gpu else "cpu"; AR = torch.arange(a.crop, device=DD)
print(f"{train['n']} train / {val['n']} val pairs on the GPU", flush=True)


def batch(D, B, gen=None):
    i = torch.randint(0, D["n"], (B,), device=DD, generator=gen)
    o = D["start"][i] + (torch.rand(B, device=DD, generator=gen) * (D["len"][i] - a.crop + 1)).long(); idx = o[:, None] + AR[None]
    to = lambda x: x.to(dev, non_blocking=True)
    return to(D["src"][:, idx].permute(1, 0, 2)).float(), to(D["tgt"][:, idx].permute(1, 0, 2)).float(), to(D["w"][idx]).float(), to(D["spk"][i])


class Disc(nn.Module):                            # patch discriminator over the log-mel image (freq x time), and a 2x-downsampled copy
    def __init__(self, ch=64):
        super().__init__()
        def stack():
            sn = nn.utils.spectral_norm
            return nn.ModuleList([sn(nn.Conv2d(1, ch, (3, 9), padding=(1, 4))), sn(nn.Conv2d(ch, ch, (3, 9), stride=(2, 1), padding=(1, 4))),
                                  sn(nn.Conv2d(ch, 2 * ch, (3, 9), stride=(2, 1), padding=(1, 4))), sn(nn.Conv2d(2 * ch, 2 * ch, (3, 3), stride=(2, 2), padding=1)),
                                  sn(nn.Conv2d(2 * ch, 4 * ch, (3, 3), padding=1)), sn(nn.Conv2d(4 * ch, 1, (3, 3), padding=1))])
        self.d = nn.ModuleList([stack(), stack()])

    def forward(self, mel):                       # mel: B, 128, T (scaled)
        outs, feats = [], []
        for k, st in enumerate(self.d):
            x = mel[:, None] if k == 0 else F.avg_pool2d(mel[:, None], 2)
            f = []
            for i, l in enumerate(st):
                x = l(x)
                if i < len(st) - 1:
                    x = F.leaky_relu(x, 0.2); f.append(x)
            outs.append(x); feats.append(f)
        return outs, feats


sc = lambda m: (m + 5.0) / 2.5
G = Mapper(d=a.d, layers=a.layers, kernel=a.kernel, attn_every=a.attn_every).to(dev)
E = Mapper(d=a.d, layers=a.layers, kernel=a.kernel, attn_every=a.attn_every).to(dev).eval(); E.load_state_dict(G.state_dict())
for p in E.parameters():
    p.requires_grad_(False)
Dn = Disc().to(dev)
print(f"mapper {sum(p.numel() for p in G.parameters()) / 1e6:.1f} M, disc {sum(p.numel() for p in Dn.parameters()) / 1e6:.1f} M", flush=True)
og = torch.optim.AdamW(G.parameters(), lr=a.lr, betas=(0.8, 0.99), weight_decay=0.01, fused=True)
od = torch.optim.AdamW(Dn.parameters(), lr=a.lr_d, betas=(0.8, 0.99), fused=True)
step = 0
if (OUT / "last.pt").exists():
    s = torch.load(OUT / "last.pt", map_location=dev, weights_only=False)
    G.load_state_dict(s["G"]); E.load_state_dict(s["E"]); Dn.load_state_dict(s["D"]); og.load_state_dict(s["og"]); od.load_state_dict(s["od"]); step = s["step"]
    print("resumed at", step, flush=True)
AMP = torch.autocast("cuda", dtype=torch.bfloat16)


@torch.no_grad()
def evaluate():
    g = torch.Generator(device=DD); g.manual_seed(0); tot = n = 0.0
    for _ in range(8):
        src, tgt, w, spk = batch(val, 32, g)
        with AMP:
            p = E(src, spk).float()
        e = (p - tgt).abs().mean(1); tot += (e * w).sum().item(); n += w.sum().item()
    return tot / n


log = open(OUT / "metrics.jsonl", "a"); t0 = time.time(); acc = {}
while step < a.steps:
    src, tgt, w, spk = batch(train, a.batch)
    adv = step >= a.l1_steps
    with AMP:
        pred = G(src, spk)
    pred = pred.float()
    l1 = ((pred - tgt).abs().mean(1) * w).sum() / w.sum().clamp_min(1)
    if adv:
        od.zero_grad(set_to_none=True)
        with AMP:
            r, _ = Dn(sc(tgt)); f_, _ = Dn(sc(pred.detach()))
        ld = sum(((x.float() - 1) ** 2).mean() + (y.float() ** 2).mean() for x, y in zip(r, f_))
        ld.backward(); od.step()
        with AMP:
            f_, ff = Dn(sc(pred)); _, rf = Dn(sc(tgt))
        lg = sum(((x.float() - 1) ** 2).mean() for x in f_)
        lfm = sum(F.l1_loss(a_.float(), b_.float().detach()) for fa, fb in zip(ff, rf) for a_, b_ in zip(fa, fb)) / sum(len(x) for x in ff)
        loss = a.lambda_l1 * l1 + lg + a.lambda_fm * lfm
    else:
        ld = lg = lfm = torch.zeros(()); loss = l1
    og.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0); og.step()
    with torch.no_grad():
        torch._foreach_lerp_(list(E.parameters()), list(G.parameters()), 1 - a.ema)
    step += 1
    for k, v in (("l1", l1), ("g", lg), ("fm", lfm), ("d", ld)):
        acc[k] = acc.get(k, 0) + float(v)
    if step % 200 == 0:
        row = {"step": step, **{k: round(v / 200, 4) for k, v in acc.items()}, "s_per_step": round((time.time() - t0) / 200, 4),
               "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        if step % a.val_every == 0:
            row["val_l1"] = round(evaluate(), 4)
        log.write(json.dumps(row) + "\n"); log.flush(); print(row, flush=True); acc = {}; t0 = time.time()
    if step % a.save_every == 0 or step == a.steps:
        torch.save({"model": E.state_dict(), "cfg": E.cfg, "step": step}, OUT / f"m_{step:06d}.pt")
    stop = (OUT / "STOP").exists()
    if step % 2000 == 0 or step == a.steps or stop:
        torch.save({"G": G.state_dict(), "E": E.state_dict(), "D": Dn.state_dict(), "og": og.state_dict(), "od": od.state_dict(), "step": step}, OUT / "last.pt.tmp")
        os.replace(OUT / "last.pt.tmp", OUT / "last.pt")
    if stop:
        (OUT / "STOP").unlink(); print("STOP file seen at", step, flush=True); raise SystemExit(0)
print("done", step)
