"""Train the mel mapper (pie/sts/model.py) on the corpus v2 pairs (pie/sts/prep.py): L1 on log-mel, random crops.
Resumable (OUT/last.pt), logs OUT/metrics.jsonl (train and val L1), keeps OUT/best.pt by val L1.
    python pie/sts/train.py OUT [--steps 40000] [--layers 8] [--kernel 7] [--d 384] [--batch 32] [--crop 256]"""
import argparse, glob, json, os, random, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from model import Mapper

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--steps", type=int, default=40000); ap.add_argument("--layers", type=int, default=8)
ap.add_argument("--kernel", type=int, default=7); ap.add_argument("--d", type=int, default=384); ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--crop", type=int, default=256); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--val_every", type=int, default=1000)
a = ap.parse_args()
torch.cuda.set_per_process_memory_fraction(0.5, 0)
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
SPK = {t: i for i, t in enumerate(C.TAGS)}


def load(split):
    items = []
    for f in sorted(glob.glob(str(C.DATA / split / "*.npz"))):
        z = np.load(f); tag = Path(f).name.split("__")[0]
        items.append((torch.from_numpy(z["src"].astype(np.float32)), torch.from_numpy(z["tgt"].astype(np.float32)), SPK[tag]))
    return items


train, val = load("train"), load("val")
print(len(train), "train,", len(val), "val pairs", flush=True)
dev = "cuda"
model = Mapper(layers=a.layers, kernel=a.kernel, d=a.d).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
step, best = 0, 1e9
if (OUT / "last.pt").exists():
    sd = torch.load(OUT / "last.pt", map_location=dev, weights_only=False)
    model.load_state_dict(sd["model"]); opt.load_state_dict(sd["opt"]); sched.load_state_dict(sd["sched"]); step, best = sd["step"], sd["best"]
    print("resumed at", step, flush=True)
log = open(OUT / "metrics.jsonl", "a")
print(f"receptive field ±{model.frames_each_side()} frames (±{model.frames_each_side() * C.HOP / C.SR * 1000:.0f} ms); "
      f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters", flush=True)


def batch():
    xs, ys, ss = [], [], []
    for _ in range(a.batch):
        s, t, k = random.choice(train)
        n = s.shape[1]; o = random.randint(0, max(0, n - a.crop))
        xs.append(s[:, o:o + a.crop]); ys.append(t[:, o:o + a.crop]); ss.append(k)
    L = max(x.shape[1] for x in xs)
    pad = lambda x: torch.nn.functional.pad(x, (0, L - x.shape[1]), value=-11.5)
    return torch.stack([pad(x) for x in xs]).to(dev), torch.stack([pad(y) for y in ys]).to(dev), torch.tensor(ss, device=dev)


def evaluate():
    model.eval(); tot = n = 0
    with torch.no_grad():
        for s, t, k in val:
            p = model(s[None].to(dev), torch.tensor([k], device=dev))
            tot += (p[0] - t.to(dev)).abs().mean().item(); n += 1
    model.train(); return tot / n


t0, run = time.time(), 0.0
while step < a.steps:
    x, y, s = batch()
    loss = (model(x, s) - y).abs().mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    step += 1; run += loss.item()
    if step % 200 == 0:
        print(f"step {step} L1 {run / 200:.4f} {(time.time() - t0) / 200 * 1000:.0f} ms/step", flush=True); run = 0.0; t0 = time.time()
    if step % a.val_every == 0 or step == a.steps:
        v = evaluate()
        log.write(json.dumps({"step": step, "val_l1": round(v, 4)}) + "\n"); log.flush()
        if v < best:
            best = v; torch.save({"model": model.state_dict(), "cfg": model.cfg, "step": step, "val_l1": v}, OUT / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "best": best}, OUT / "last.pt.tmp")
        os.replace(OUT / "last.pt.tmp", OUT / "last.pt")
        print(f"val L1 {v:.4f} (best {best:.4f})", flush=True)
print("done", step)
