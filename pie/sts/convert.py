"""Convert Chimera renders with a trained mapper (pie/sts/train.py OUT/best.pt, over the released BigVGAN-v2) or an
end-to-end checkpoint (pie/sts/train_e2e.py OUT/g_<step>.pt, mapper and fine-tuned generator).
    python pie/sts/convert.py CKPT OUT [--src DIR] [--split val]
Without --src: the val pairs' source mels. With --src DIR (renders named <clip>__<JD3-PIE|W02-PIE>.wav): those
renders, mels computed here. Each val clip is converted into the voice(s) whose val list holds it.
-> OUT/<TAG>/<stem>.wav (44.1 kHz, 16-bit)"""
import argparse, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from model import Mapper

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("out"); ap.add_argument("--src"); ap.add_argument("--split", default="val")
a = ap.parse_args()
torch.cuda.set_per_process_memory_fraction(0.5, 0)
sd = torch.load(a.ckpt, map_location="cuda", weights_only=False)        # our own checkpoints
if "gen" in sd:                                    # an end-to-end checkpoint (pie/sts/train_e2e.py): mapper + generator
    m = Mapper(**sd["mapper_cfg"]).cuda().eval(); m.load_state_dict(sd["mapper"])
    voc = C.vocoder(weight_norm=True); voc.load_state_dict(sd["gen"]); voc.remove_weight_norm(); voc.eval()
else:                                              # an L1 mapper (pie/sts/train.py) over the released vocoder
    m = Mapper(**sd["cfg"]).cuda().eval(); m.load_state_dict(sd["model"])
    voc = C.vocoder()
SPK = {t: i for i, t in enumerate(C.TAGS)}
n = 0
for tag, stem in C.pairs(a.split):
    if a.src:
        p = Path(a.src) / f"{stem}.wav"
        if not p.exists():
            continue
        y, sr = sf.read(p, dtype="float32"); assert sr == C.SR
        src = C.mel(y).cuda()
    else:
        f = C.DATA / a.split / f"{tag}__{stem}.npz"
        if not f.exists():
            continue
        src = torch.from_numpy(np.load(f)["src"].astype(np.float32)).cuda()
    with torch.no_grad():
        mel = m(src[None], torch.tensor([SPK[tag]], device="cuda"))
        w = voc(mel)[0, 0].float().cpu().numpy()
    d = Path(a.out) / tag; d.mkdir(parents=True, exist_ok=True)
    sf.write(d / f"{stem}.wav", np.clip(w, -1, 1), C.SR, subtype="PCM_16"); n += 1
print(n, "clips ->", a.out)
