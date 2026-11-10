"""Train v2, the flow-matching converter (pie/sts/flow.py), on the corpus v2 pairs (pie/sts/prep.py), the vocoder out of
the loop entirely (it is the released BigVGAN-v2, used only at conversion).

Speed: every pair's source and target mels live on the GPU (fp16, ~2.5 GB), and so does each pair's frame weight;
a batch is gathered there with one index op, every crop is the full CROP frames (a short clip is padded with log-mel
silence on both sides, a valid silence -> silence pair, so attention never needs a mask and runs on the flash kernel),
bf16 autocast, torch.compile of the network, fused AdamW, EMA of the weights for evaluation and conversion.

The accuracy mask (--mask, pie/sts/mask.py): frames whose centre falls in a masked interval (the sounds the paid target
gets wrong more often than the Chimera render) carry no loss, so the model is not taught the target's error there; it
still has to produce a natural frame, since every other frame of the same voice teaches it what one is.

Resumable: OUT/last.pt every --last_every steps; OUT/STOP saves and exits 0; OUT/ema_<step>.pt every --save_every
steps (EMA weights, for pie/sts/convert.py); OUT/metrics.jsonl (train loss, val loss at fixed noise and times).
    python pie/sts/train_flow.py OUT [--steps 120000] [--batch 64] [--crop 256] [--mask mask.json]
"""
import argparse, glob, json, math, os, random, sys, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C
from flow import FlowNet, scale, SIGMA_MIN

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--steps", type=int, default=120000); ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--crop", type=int, default=256); ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--warmup", type=int, default=2000)
ap.add_argument("--d", type=int, default=512); ap.add_argument("--layers", type=int, default=12); ap.add_argument("--heads", type=int, default=8)
ap.add_argument("--p_uncond", type=float, default=0.1); ap.add_argument("--ema", type=float, default=0.9999)
ap.add_argument("--mask", default=""); ap.add_argument("--mem", type=float, default=0.78)
ap.add_argument("--save_every", type=int, default=10000); ap.add_argument("--last_every", type=int, default=2000)
ap.add_argument("--val_every", type=int, default=2000); ap.add_argument("--no_compile", action="store_true")
ap.add_argument("--bench", type=int, default=0, help="run this many steps, print ms/step and peak memory, exit")
a = ap.parse_args()
dev = "cuda"
torch.cuda.set_per_process_memory_fraction(a.mem, 0)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
SPK = {t: i for i, t in enumerate(C.TAGS)}
SIL = -11.5                                                  # log-mel silence (the clamp at 1e-5)
MASKS = json.load(open(a.mask)) if a.mask else {}


def frame_weight(key, n):                                    # 1 per frame, 0 where the frame's centre is masked
    w = np.ones(n, np.float16)
    centres = np.arange(n) * C.HOP + 1024                    # BigVGAN mels: window 2048, hop 512, not centred
    for s0, s1 in MASKS.get(key, []):
        w[(centres >= s0) & (centres < s1)] = 0
    return w


def load(split):
    srcs, tgts, ws, spk, lens = [], [], [], [], []
    for f in sorted(glob.glob(str(C.DATA / split / "*.npz"))):
        z = np.load(f); tag, stem = Path(f).stem.split("__", 1)
        n = z["src"].shape[1]
        srcs.append(z["src"]); tgts.append(z["tgt"]); ws.append(frame_weight(f"{tag}/{stem}", n)); spk.append(SPK[tag]); lens.append(n)
    # one flat GPU buffer per kind (time-major), each clip padded on the right to at least CROP with silence (weight 1)
    starts, cur = [], 0
    S, T, W = [], [], []
    for s, t, w, n in zip(srcs, tgts, ws, lens):
        m = max(n, a.crop)
        S.append(np.pad(s, ((0, 0), (0, m - n)), constant_values=SIL)); T.append(np.pad(t, ((0, 0), (0, m - n)), constant_values=SIL))
        W.append(np.pad(w, (0, m - n), constant_values=1)); starts.append(cur); cur += m
    g = lambda xs: torch.from_numpy(np.concatenate(xs, 1 if xs[0].ndim == 2 else 0)).to(dev)
    return dict(src=g(S), tgt=g(T), w=g(W), start=torch.tensor(starts, device=dev), len=torch.tensor([max(n, a.crop) for n in lens], device=dev),
                spk=torch.tensor(spk, device=dev), n=len(lens), masked=float(1 - np.concatenate(ws).astype(np.float32).mean()))


train, val = load("train"), load("val")
print(f"{train['n']} train / {val['n']} val pairs on the GPU; {100 * train['masked']:.1f} % of training frames masked", flush=True)
AR = torch.arange(a.crop, device=dev)


def batch(D, B, gen=None):
    i = torch.randint(0, D["n"], (B,), device=dev, generator=gen)
    o = D["start"][i] + (torch.rand(B, device=dev, generator=gen) * (D["len"][i] - a.crop + 1)).long()
    idx = o[:, None] + AR[None]                                           # B, CROP: flat frame indices
    src = D["src"][:, idx].permute(1, 0, 2).float(); tgt = D["tgt"][:, idx].permute(1, 0, 2).float()
    return scale(src), scale(tgt), D["w"][idx].float(), D["spk"][i]


net = FlowNet(d=a.d, layers=a.layers, heads=a.heads).to(dev)
ema = FlowNet(d=a.d, layers=a.layers, heads=a.heads).to(dev).eval(); ema.load_state_dict(net.state_dict())
for p in ema.parameters():
    p.requires_grad_(False)
print(f"{sum(p.numel() for p in net.parameters()) / 1e6:.1f} M parameters", flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.01, fused=True)
lr_at = lambda s: a.lr * min(1, (s + 1) / a.warmup) * (0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))) * 0.9 + 0.1)
step = 0
if (OUT / "last.pt").exists():
    sd = torch.load(OUT / "last.pt", map_location=dev, weights_only=False)
    net.load_state_dict(sd["net"]); ema.load_state_dict(sd["ema"]); opt.load_state_dict(sd["opt"]); step = sd["step"]
    print("resumed at", step, flush=True)
fwd = net if a.no_compile else torch.compile(net)


def loss_fn(model, src, tgt, w, spk, t, x0):
    xt = (1 - (1 - SIGMA_MIN) * t[:, None, None]) * x0 + t[:, None, None] * tgt
    u = tgt - (1 - SIGMA_MIN) * x0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        v = model(xt, src, spk, t)
    e = ((v.float() - u) ** 2).mean(1)                                    # B, T
    return (e * w).sum() / w.sum().clamp_min(1)


@torch.no_grad()
def evaluate():
    g = torch.Generator(device=dev); g.manual_seed(0); tot = 0.0
    for k in range(8):                                                    # fixed crops, noise and times: comparable
        src, tgt, w, spk = batch(val, 64, g)
        t = (torch.arange(64, device=dev) + 0.5) / 64
        x0 = torch.randn(tgt.shape, device=dev, generator=g)
        tot += loss_fn(ema, src, tgt, w, spk, t, x0).item()
    return tot / 8


def save_last():
    torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(a)}, OUT / "last.pt.tmp")
    os.replace(OUT / "last.pt.tmp", OUT / "last.pt")


log = open(OUT / "metrics.jsonl", "a")
t0, run, n_run = time.time(), 0.0, 0
torch.cuda.reset_peak_memory_stats()
while step < a.steps:
    for gr in opt.param_groups:
        gr["lr"] = lr_at(step)
    src, tgt, w, spk = batch(train, a.batch)
    drop = torch.rand(a.batch, device=dev) < a.p_uncond                  # classifier-free guidance: no source, no voice
    src = torch.where(drop[:, None, None], torch.zeros_like(src), src); spk = torch.where(drop, torch.full_like(spk, len(C.TAGS)), spk)
    t = torch.sigmoid(torch.randn(a.batch, device=dev))                   # logit-normal flow times
    x0 = torch.randn_like(tgt)
    loss = loss_fn(fwd, src, tgt, w, spk, t, x0)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    with torch.no_grad():
        torch._foreach_lerp_(list(ema.parameters()), list(net.parameters()), 1 - (a.ema if step > a.warmup else 0.99))
    step += 1; run += loss.item() if step % 50 == 0 else 0; n_run += step % 50 == 0
    if a.bench and step == a.bench:
        torch.cuda.synchronize(); dt = (time.time() - t0) / a.bench
        print(f"BENCH batch {a.batch} x {a.crop}: {dt * 1000:.0f} ms/step (incl. compile), peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        t1 = time.time()
        for _ in range(20):
            src, tgt, w, spk = batch(train, a.batch); t = torch.rand(a.batch, device=dev)
            l = loss_fn(fwd, src, tgt, w, spk, t, torch.randn_like(tgt)); opt.zero_grad(); l.backward(); opt.step()
        torch.cuda.synchronize(); print(f"BENCH steady: {(time.time() - t1) / 20 * 1000:.0f} ms/step", flush=True)
        raise SystemExit(0)
    if step % 500 == 0:
        row = {"step": step, "loss": round(run / max(1, n_run), 4), "lr": round(lr_at(step), 7), "s_per_step": round((time.time() - t0) / 500, 4),
               "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        if step % a.val_every == 0:
            row["val"] = round(evaluate(), 4)
        log.write(json.dumps(row) + "\n"); log.flush(); print(row, flush=True); run, n_run, t0 = 0.0, 0, time.time()
    if step % a.save_every == 0 or step == a.steps:
        torch.save({"ema": ema.state_dict(), "cfg": net.cfg, "step": step, "kind": "flow"}, OUT / f"ema_{step:06d}.pt")
    stop = (OUT / "STOP").exists()
    if step % a.last_every == 0 or step == a.steps or stop:
        save_last()
    if stop:
        (OUT / "STOP").unlink(); print("STOP file seen: saved last.pt at step", step, flush=True); raise SystemExit(0)
print("done", step)
