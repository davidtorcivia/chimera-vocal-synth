"""v1: the mapper (pie/sts/model.py, from its L1 checkpoint) and BigVGAN-v2's generator fine-tuned end to end on the
corpus v2 pairs, against the restored target waveform, with BigVGAN's own training losses: its multi-scale mel L1
(x15), and its pretrained discriminators (multi-period, multi-scale sub-band CQT) for the adversarial and
feature-matching terms. The L1-only mapper made smooth mels the frozen vocoder turned to grit; here the vocoder learns
to read the mapper's mels and both are pushed toward real speech.

Segments of SEG mel frames (512 samples each) cut at random; the target waveform segment is the same frames' samples.
Resumable: OUT/last.pt every --last_every steps (500, ~8 min), and a start with it present resumes. Stop cleanly by
creating OUT/STOP: it saves last.pt at the next step and exits 0. OUT/metrics.jsonl; OUT/g_<step>.pt every --save_every
steps (mapper + generator, for conversion).
    python pie/sts/train_e2e.py OUT --init MAPPER.pt [--steps 30000] [--batch 4] [--seg 128] [--d_warmup 500]
"""
import argparse, glob, itertools, json, os, random, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C
REPO = C.ROOT / "BigVGAN-repo"                       # NVIDIA/BigVGAN @ 7d2b454 (discriminators, losses)
sys.path.insert(0, str(REPO))
from env import AttrDict                                          # noqa: E402
import bigvgan                                                    # noqa: E402
from discriminators import MultiPeriodDiscriminator, MultiScaleSubbandCQTDiscriminator   # noqa: E402
from loss import MultiScaleMelSpectrogramLoss, feature_loss, generator_loss, discriminator_loss   # noqa: E402
from model import Mapper                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--init", required=True); ap.add_argument("--steps", type=int, default=30000)
ap.add_argument("--batch", type=int, default=4); ap.add_argument("--seg", type=int, default=128)
ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--lr_map", type=float, default=1e-4)
ap.add_argument("--d_warmup", type=int, default=500); ap.add_argument("--save_every", type=int, default=2500)
ap.add_argument("--mem", type=float, default=0.6)
ap.add_argument("--last_every", type=int, default=500, help="resume point (last.pt) every this many steps")
ap.add_argument("--amp", default="", help="bf16: autocast the mapper and generator (gen), or also the discriminators (all)")
ap.add_argument("--compile", default="", help="comma list of gen,mpd,mrd,mel: torch.compile them (segment shapes are fixed)")
ap.add_argument("--no_ckpt", action="store_true", help="no activation checkpointing on the generator's residual blocks")
ap.add_argument("--bench", type=int, default=0, help="run this many steps, print a timing breakdown, exit (nothing saved)")
a = ap.parse_args()
torch.cuda.set_per_process_memory_fraction(a.mem, 0)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"
h = AttrDict(json.load(open(C.BIGVGAN / "config.json")))
SPK = {t: i for i, t in enumerate(C.TAGS)}

# data: source mels and the target waveforms (int16 in RAM), frame-aligned
items = []
for f in sorted(glob.glob(str(C.DATA / "train" / "*.npz"))):
    tag, stem = Path(f).stem.split("__", 1)
    src = np.load(f)["src"]
    w, sr = sf.read(C.RESTORED / tag / f"{stem}.wav", dtype="int16")
    n = src.shape[1] * C.HOP
    w = np.pad(w, (0, max(0, n - len(w))))[:n]
    items.append((src, w, SPK[tag]))
print(len(items), "training pairs in memory", flush=True)


def batch():
    xs, ys, ss = [], [], []
    for _ in range(a.batch):
        src, w, k = random.choice(items)
        T = src.shape[1]
        o = random.randint(0, max(0, T - a.seg))
        m = src[:, o:o + a.seg].astype(np.float32); y = w[o * C.HOP:(o + a.seg) * C.HOP].astype(np.float32) / 32768
        if m.shape[1] < a.seg:                                  # short clips: pad with silence
            m = np.pad(m, ((0, 0), (0, a.seg - m.shape[1])), constant_values=-11.5)
            y = np.pad(y, (0, a.seg * C.HOP - len(y)))
        xs.append(m); ys.append(y); ss.append(k)
    return (torch.from_numpy(np.stack(xs)).to(dev), torch.from_numpy(np.stack(ys))[:, None].to(dev), torch.tensor(ss, device=dev))


sd = torch.load(a.init, map_location="cpu", weights_only=False)        # our own L1 checkpoint
mapper = Mapper(**sd["cfg"]).to(dev); mapper.load_state_dict(sd["model"])
gen = bigvgan.BigVGAN(h, use_cuda_kernel=False).to(dev)
gen.load_state_dict(torch.load(C.BIGVGAN / "bigvgan_generator.pt", map_location="cpu", weights_only=True)["generator"])
# activation checkpointing on the generator's residual blocks (their anti-aliased activations hold most of the memory;
# BigVGAN trained 4 x 65536-sample segments per 80 GB card). Exact: the generator draws no random numbers.
from torch.utils.checkpoint import checkpoint
if not a.no_ckpt:
    for rb in gen.resblocks:
        rb.forward = (lambda f: (lambda x: checkpoint(f, x, use_reentrant=False)))(rb.forward)
mpd, mrd = MultiPeriodDiscriminator(h).to(dev), MultiScaleSubbandCQTDiscriminator(h).to(dev)
dsd = torch.load(C.BIGVGAN / "bigvgan_discriminator_optimizer.pt", map_location="cpu", weights_only=True)
mpd.load_state_dict(dsd["mpd"]); mrd.load_state_dict(dsd["mrd"]); del dsd
mel_loss = MultiScaleMelSpectrogramLoss(sampling_rate=h.sampling_rate).to(dev)
for name in [c for c in a.compile.split(",") if c]:
    {"gen": gen, "mpd": mpd, "mrd": mrd, "mel": mel_loss}[name].compile()       # in place: state_dict keys unchanged
AMP_G = torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp in ("gen", "all"))
AMP_D = torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp == "all")
EV = {}


def mark(k):
    if a.bench:
        e = torch.cuda.Event(enable_timing=True); e.record(); EV.setdefault(k, []).append(e)
og = torch.optim.AdamW([{"params": gen.parameters(), "lr": a.lr}, {"params": mapper.parameters(), "lr": a.lr_map}], betas=(h.adam_b1, h.adam_b2))
od = torch.optim.AdamW(itertools.chain(mpd.parameters(), mrd.parameters()), a.lr, betas=(h.adam_b1, h.adam_b2))
step = 0
if (OUT / "last.pt").exists():
    s = torch.load(OUT / "last.pt", map_location=dev, weights_only=False)
    mapper.load_state_dict(s["mapper"]); gen.load_state_dict(s["gen"]); mpd.load_state_dict(s["mpd"]); mrd.load_state_dict(s["mrd"])
    og.load_state_dict(s["og"]); od.load_state_dict(s["od"]); step = s["step"]
    print("resumed at", step, flush=True)
log = open(OUT / "metrics.jsonl", "a")
clip = h.get("clip_grad_norm", 500)


def save_last():
    torch.save({"mapper": mapper.state_dict(), "gen": gen.state_dict(), "mpd": mpd.state_dict(), "mrd": mrd.state_dict(),
                "og": og.state_dict(), "od": od.state_dict(), "step": step}, OUT / "last.pt.tmp")
    os.replace(OUT / "last.pt.tmp", OUT / "last.pt")


def save(path):
    torch.save({"mapper": mapper.state_dict(), "mapper_cfg": mapper.cfg, "gen": gen.state_dict(), "step": step}, path)


t0, acc = time.time(), {}
mapper.train(); gen.train(); mpd.train(); mrd.train()
while step < a.steps:
    x, y, s = batch()
    mark("0 start")
    with AMP_G:
        y_hat = gen(mapper(x, s))
    y_hat = y_hat.float()
    mark("1 gen fwd")
    # discriminators
    od.zero_grad()
    with AMP_D:
        r1, g1, _, _ = mpd(y, y_hat.detach()); ld_f, _, _ = discriminator_loss(r1, g1)
        r2, g2, _, _ = mrd(y, y_hat.detach()); ld_s, _, _ = discriminator_loss(r2, g2)
    ld = ld_f + ld_s
    ld.backward(); torch.nn.utils.clip_grad_norm_(itertools.chain(mpd.parameters(), mrd.parameters()), clip); od.step()
    mark("2 disc step")
    # generator + mapper
    og.zero_grad()
    lm = mel_loss(y, y_hat) * h.lambda_melloss
    mark("3 mel loss")
    if step >= a.d_warmup or a.bench:
        with AMP_D:
            _, g1, f1r, f1g = mpd(y, y_hat); _, g2, f2r, f2g = mrd(y, y_hat)
        lg = generator_loss(g1)[0] + generator_loss(g2)[0]; lfm = feature_loss(f1r, f1g) + feature_loss(f2r, f2g)
        lall = lm + lg + lfm
    else:
        lg = lfm = torch.zeros(())
        lall = lm
    mark("4 disc on fake")
    lall.backward(); torch.nn.utils.clip_grad_norm_(itertools.chain(gen.parameters(), mapper.parameters()), clip); og.step()
    mark("5 gen backward")
    step += 1
    if a.bench and step == a.bench:
        torch.cuda.synchronize()
        ks = sorted(EV); w = 5                                          # skip the first steps (compile, warm-up)
        per = {ks[i]: np.mean([EV[ks[i - 1]][j].elapsed_time(EV[ks[i]][j]) for j in range(w, a.bench)]) for i in range(1, len(ks))}
        tot = np.mean([EV[ks[0]][j].elapsed_time(EV[ks[-1]][j]) for j in range(w, a.bench)])
        print(f"BENCH amp={a.amp or 'off'} compile={a.compile or 'off'} ckpt={'off' if a.no_ckpt else 'on'} batch={a.batch} seg={a.seg}: "
              f"{tot:.0f} ms/step, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB | "
              + ", ".join(f"{k[2:]} {v:.0f}" for k, v in per.items()), flush=True)
        raise SystemExit(0)
    for k, v in (("mel", lm.item() / h.lambda_melloss), ("g", float(lg)), ("fm", float(lfm)), ("d", ld.item())):
        acc[k] = acc.get(k, 0) + v
    if step % 100 == 0:
        row = {"step": step, **{k: round(v / 100, 4) for k, v in acc.items()}, "s_per_step": round((time.time() - t0) / 100, 3),
               "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        log.write(json.dumps(row) + "\n"); log.flush(); print(row, flush=True); acc = {}; t0 = time.time()
    if step % a.save_every == 0 or step == a.steps:
        save(OUT / f"g_{step:06d}.pt")
    stop = (OUT / "STOP").exists()
    if step % a.last_every == 0 or step == a.steps or stop:
        save_last()
    if stop:
        (OUT / "STOP").unlink()
        print("STOP file seen: saved last.pt at step", step, flush=True)
        log.write(json.dumps({"event": "stopped", "step": step}) + "\n"); log.flush()
        raise SystemExit(0)
print("done", step)
