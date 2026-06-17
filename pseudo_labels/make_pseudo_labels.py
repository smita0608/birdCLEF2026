import os, gc, numpy as np, pandas as pd, soundfile as sf
from pathlib import Path
import onnxruntime as ort
import torch, torchaudio

BASE         = Path("/kaggle/input/competitions/birdclef-2026")
MODEL_DIR    = Path("/kaggle/input/datasets/smitapriyadarshani/birdclef2026-cnn-v2-fixed-models")
OUT_PARQUET  = Path("/kaggle/working/pseudo_labels.parquet")

SR, CLIP_SEC = 32000, 5.0
WIN_SAMPLES  = int(SR * CLIP_SEC)
N_FFT, HOP, N_MELS = 2048, 512, 128
FMIN, FMAX   = 50, 14000
FILE_SEC     = 60
N_WIN        = 12

TOPK            = 5
CONF_THRESHOLD  = 0.3
SOFT_FLOOR      = 0.02

sample_sub = pd.read_csv(BASE / "sample_submission.csv")
CLASSES = sample_sub.columns[1:].tolist()
N_CLASSES = len(CLASSES)

mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
    f_min=FMIN, f_max=FMAX, power=2.0)
db = torchaudio.transforms.AmplitudeToDB(top_db=80)

def wav_to_mel(wav):
    x = torch.from_numpy(wav.astype(np.float32))
    m = db(mel(x))
    m = (m - m.mean()) / (m.std() + 1e-6)
    return m.unsqueeze(0).unsqueeze(0).numpy().astype(np.float32)

sessions = []
for p in sorted(MODEL_DIR.glob("*.onnx")):
    sessions.append(ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]))
    print("loaded:", p.name)
print(f"ensemble size: {len(sessions)}")

rows = []
files = sorted((BASE / "train_soundscapes").glob("*.ogg"))
print(f"soundscapes: {len(files)}")

for fi, fp in enumerate(files):
    try:
        wav, _sr = sf.read(str(fp), dtype="float32", always_2d=False)
        if wav.ndim == 2: wav = wav.mean(axis=1)
        need = FILE_SEC * SR
        if len(wav) < need: wav = np.pad(wav, (0, need - len(wav)))
        else: wav = wav[:need]
    except Exception as e:
        print("skip", fp.name, e); continue

    for wi in range(N_WIN):
        s = wi * WIN_SAMPLES
        chunk = wav[s:s + WIN_SAMPLES]
        m = wav_to_mel(chunk)
        probs = []
        for sess in sessions:
            name = sess.get_inputs()[0].name
            p = sess.run(None, {name: m})[0][0]
            probs.append(p)
        p_mean = np.mean(probs, axis=0).astype(np.float32)

        order = np.argsort(-p_mean)
        keep = set(order[:TOPK].tolist())
        y = np.full(N_CLASSES, SOFT_FLOOR, dtype=np.float32)
        for k in keep:
            if p_mean[k] >= CONF_THRESHOLD:
                y[k] = float(p_mean[k])

        rows.append({"filename": fp.name, "start_sec": s // SR, "labels": y})

    if (fi + 1) % 20 == 0:
        print(f"  {fi+1}/{len(files)} files")

pdf = pd.DataFrame(rows)
pdf["labels"] = pdf["labels"].apply(lambda a: a.tolist())
pdf.to_parquet(OUT_PARQUET, index=False)
print(f"wrote {OUT_PARQUET}  rows={len(pdf)}")
