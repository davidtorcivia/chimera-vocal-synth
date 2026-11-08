#!/usr/bin/env python3
"""Quick probe: render short PIE items with `chimera render pie` and print what PhoneticXeus hears.

    python pie/probe.py "ádʰa ‖ áda ‖ áḱa" [male|female] [option=value ...]
Prints each phrase's greedy transcription and, for the middle consonant of a VCV item, the top-5 non-blank
tokens with their probabilities at the frame where it is most confident.
"""
import os, subprocess, sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gop as G
import onnxruntime as ort

BIN = Path(__file__).resolve().parent.parent / os.environ.get("CHIMERA_BIN", "target/release/chimera")
OUT = G.E.DATA / "acoustic" / "probe"


def main():
    text = sys.argv[1]
    body = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in ("male", "female") else "male"
    opts = [a for a in sys.argv[2:] if "=" in a]
    r = subprocess.run([str(BIN), "render", "pie", str(OUT), text, body, *opts, "--float"], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stderr)
    marks = [(t.rsplit("@", 1)[0], float(t.rsplit("@", 1)[1])) for t in [l for l in r.stdout.split("\n") if "@" in l][0].split()]
    x, sr = sf.read(str(OUT) + ".wav", dtype="float32")
    x = resample_poly(x, 1, 3).astype(np.float32)
    sess = ort.InferenceSession(str(G.E.DATA / "hf/pxeus-convfloat.onnx"), providers=["CPUExecutionProvider"])
    starts = [k for k, (lab, _) in enumerate(marks) if "́" in lab and lab[0] in "aeiouə"] + [len(marks)]
    for a, b in zip(starts, starts[1:]):
        g = marks[a:b]
        t0 = g[0][1] - 0.08
        t1 = (marks[b][1] - 0.25) if b < len(marks) else g[-1][1] + 0.2
        seg = x[int(t0 * 16000):int(t1 * 16000)]
        lp = G.logits(sess, seg)
        ids = lp.argmax(1)
        out, prev = [], -1
        for i in ids:
            if i != prev and i not in G.SPECIAL:
                out.append(G.ID2TOK[int(i)])
            prev = i
        label = "".join(l for l, _ in g)
        line = f"{label:12} heard: {' '.join(out)}"
        if len(g) >= 3:
            hop = len(seg) / lp.shape[0] / 16000
            ca, cb = int((g[1][1] - t0) / hop), int((g[-1][1] - t0 + 0.03) / hop) + 1
            p = np.exp(lp[ca:cb])
            p[:, list(G.SPECIAL)] = 0
            k = int(np.argmax(p.sum(1)))
            top = np.argsort(-p[k])[:5]
            line += "   | C: " + " ".join(f"{G.ID2TOK[int(t)]} {p[k, t] / p[k].sum():.2f}" for t in top)
            mur = [i for t, i in G.VOCAB.items() if (t.endswith("ʰ") and t[0] in "bdɡɟ") or "\u0324" in t]
            asp = [i for t, i in G.VOCAB.items() if t.endswith("ʰ") and t[0] in "ptkcq"]
            line += f"   [murmured {p[:, mur].sum(1).max() / max(1e-9, p.sum(1)[np.argmax(p[:, mur].sum(1))]):.2f}]"
        print(line)


if __name__ == "__main__":
    main()
