#!/usr/bin/env python3
"""Export corpus v2 (restored, QC-passed clips) as a Kokoro/StyleTTS2 training set, in the PIE repo's corpus format
(tts/vtl/corpus_pipeline.py `final`): 24 kHz mono 16-bit WAV (soxr_vhq, peak capped at 0.99), train_list.txt and
val_list.txt with FILE|LABEL|SPEAKER, labels and the train/val split from corpus v1's manifest (the same sentences),
rejected.jsonl and summary.json.

    python pie/export_kokoro.py QC.json [--workers 12]
QC.json: {"TAG/stem": {"pass": bool, "why": [...]}, ...} (pie/gate_v2.py).
"""
import argparse, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

FULL = Path.home() / "AI/chimera-pie-data/full"
OUT = FULL / "final-v2"
SPEAKER = {"ZAZY": "pie_zazy", "ANTONIO": "pie_antonio", "VALERIA": "pie_valeria", "W02": "pie_w02"}


def final(job):
    import librosa, numpy as np, soundfile as sf
    src, dst = job
    y, sr = librosa.load(str(src), sr=None, mono=True)
    y = librosa.resample(y, orig_sr=sr, target_sr=24000, res_type="soxr_vhq")
    pk = float(np.abs(y).max())
    if pk > 0.99:
        y = y / pk * 0.99
    sf.write(str(dst), y, 24000, subtype="PCM_16")
    return len(y) / 24000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qc")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    qc = json.load(open(a.qc))
    man = {}
    for l in open(FULL / "manifest.jsonl", encoding="utf-8"):
        m = json.loads(l)
        man[f"{m['clip']}__{m['speaker']}"] = m
    (OUT / "wavs").mkdir(parents=True, exist_ok=True)
    rows, rejected = [], []
    for tag in SPEAKER:
        for p in sorted((FULL / "restored-v2" / tag).glob("*.wav")):
            key = f"{tag}/{p.stem}"
            q = qc.get(key)
            if not q or not q["pass"]:
                rejected.append({"file": f"{tag}__{p.stem}.wav", "why": q["why"] if q else ["missing qc"]})
                continue
            m = man[p.stem]
            name = f"{m['clip']}__{tag}.wav"
            rows.append((p, OUT / "wavs" / name, name, m["label"], SPEAKER[tag], m["split"]))
    with ProcessPoolExecutor(a.workers) as ex:
        durs = list(ex.map(final, [(r[0], r[1]) for r in rows], chunksize=8))
    lists = {"train": [], "val": []}
    secs = {"train": {}, "val": {}}
    for (src, dst, name, label, spk, split), d in zip(rows, durs):
        lists[split].append(f"{name}|{label}|{spk}")
        secs[split][spk] = secs[split].get(spk, 0) + d
    for s in lists:
        (OUT / f"{s}_list.txt").write_text("\n".join(lists[s]) + "\n", "utf-8")
    with open(OUT / "rejected.jsonl", "w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"files": {s: len(v) for s, v in lists.items()},
               "hours": {s: {k: round(v / 3600, 3) for k, v in d.items()} for s, d in secs.items()},
               "rejected": len(rejected), "format": "WAV PCM_16 mono 24000 Hz",
               "list_format": "FILE|LABEL|SPEAKER (files relative to wavs/)",
               "source": "corpus v2: Chimera dcc37da renders → ElevenLabs sts v2 → restoration v2 (pie/restore_full.py)"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), "utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
