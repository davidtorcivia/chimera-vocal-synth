#!/usr/bin/env python3
"""Voice onset after voiceless sounds: for each C in "áCa"-type items, the time from the release (the end of
the consonant's span) to the first voiced frame (Praat pitch, 2 ms steps), and the voiced share of the
next vowel's first 50 ms. Plain voiceless stops: about the VOT target (15 ms); after s, χ, h the voice
should resume within ~10 ms.

    python pie/vot.py [male|female] [option=value ...]
"""
import os, subprocess, sys
from pathlib import Path

import numpy as np
import parselmouth

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / os.environ.get("CHIMERA_BIN", "target/release/chimera")
OUT = Path(os.environ.get("CHIMERA_PIE_DATA", Path.home() / "AI/chimera-pie-data")) / "acoustic" / "vot"
ITEMS = "ápa ‖ áta ‖ áka ‖ ása ‖ áh₂a ‖ áha ‖ pəté ‖ sésəta"


def main():
    body = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("male", "female") else "male"
    opts = [a for a in sys.argv[1:] if "=" in a]
    r = subprocess.run([str(BIN), "render", "pie", str(OUT), ITEMS, body, *opts, "--float"], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stderr)
    m = [(t.rsplit("@", 1)[0], float(t.rsplit("@", 1)[1])) for t in [l for l in r.stdout.split("\n") if "@" in l][0].split()]
    snd = parselmouth.Sound(str(OUT) + ".wav")
    pitch = snd.to_pitch(time_step=0.002, pitch_floor=70, pitch_ceiling=400)
    out = []
    for i in range(len(m) - 1):
        lab, t = m[i]
        if lab in ("p", "t", "k", "s", "χ", "h") and i + 1 < len(m) and m[i + 1][0][0] in "aeiouəáéíóú":
            rel = m[i + 1][1]
            ts = np.arange(rel - 0.01, rel + 0.08, 0.002)
            v = np.array([pitch.get_value_at_time(x) > 0 for x in ts])
            on = next((x for x, ok in zip(ts, v) if ok and x >= rel - 0.005), float("nan"))
            nxt = m[i + 2][1] if i + 2 < len(m) else rel + 0.1
            ts2 = np.arange(rel, min(nxt, rel + 0.05), 0.002)
            share = np.mean([pitch.get_value_at_time(x) > 0 for x in ts2])
            out.append(f"{lab}{m[i+1][0]}: onset {1000*(on-rel):+4.0f} ms, voiced {share*100:3.0f}%")
    print(" | ".join(out))


if __name__ == "__main__":
    main()
