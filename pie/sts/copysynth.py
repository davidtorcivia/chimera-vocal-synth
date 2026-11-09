"""The vocoder's ceiling: BigVGAN-v2 re-synthesizes the restored validation clips from their own mels (no converter).
    python pie/sts/copysynth.py OUT          -> OUT/<TAG>/<stem>.wav (44.1 kHz)"""
import sys
from pathlib import Path
import numpy as np, soundfile as sf, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

out = Path(sys.argv[1]); voc = C.vocoder()
torch.cuda.set_per_process_memory_fraction(0.5, 0)
for tag, stem in C.pairs("val"):
    d = out / tag; d.mkdir(parents=True, exist_ok=True)
    f = C.DATA / "val" / f"{tag}__{stem}.npz"
    if not f.exists():                       # dropped by prep.py (alignment)
        continue
    z = np.load(f)
    with torch.no_grad():
        y = voc(torch.from_numpy(z["tgt"].astype(np.float32))[None].cuda())[0, 0].float().cpu().numpy()
    sf.write(d / f"{stem}.wav", np.clip(y, -1, 1), C.SR, subtype="PCM_16")
print("done", out)
