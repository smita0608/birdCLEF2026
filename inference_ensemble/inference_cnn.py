import os, time
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
import onnxruntime as ort
import torch, torchaudio

SR           = 32000
CLIP_SEC     = 5.0
WIN_SAMPLES  = int(SR * CLIP_SEC)
FILE_SEC     = 60
N_WIN        = 12
N_FFT, HOP   = 2048, 512
N_MELS       = 128
FMIN, FMAX   = 50, 14000

CNN_MODEL_DIRS = [
    Path("/kaggle/input/datasets/smitapriyadarshani/birdclef2026-cnn-v2-fold0-fixed"),
]

def _build_session(path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])

_sessions = []
for d in CNN_MODEL_DIRS:
    if not d.exists():
        continue
    for p in sorted(d.glob("*.onnx")):
        _sessions.append((p.name, _build_session(p)))
print(f"[inference_cnn] Loaded {len(_sessions)} ONNX models")

_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
    f_min=FMIN, f_max=FMAX, power=2.0)
_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

def _file_to_mel_batch(wav60, shift_samples=0):
    total = len(wav60)
    chunks = np.zeros((N_WIN, WIN_SAMPLES), dtype=np.float32)
    for wi in range(N_WIN):
        s = wi * WIN_SAMPLES + shift_samples
        s = max(0, min(total - WIN_SAMPLES, s))
        chunks[wi] = wav60[s:s + WIN_SAMPLES]
    x = torch.from_numpy(chunks)
    with torch.no_grad():
        m = _db(_mel(x))
        m = (m - m.mean(dim=(-2,-1), keepdim=True)) / \
            (m.std(dim=(-2,-1), keepdim=True) + 1e-6)
    return m.unsqueeze(1).numpy().astype(np.float32)

def infer_cnn(test_paths, meta_test, n_classes, tta_shifts_sec=(0.0,), verbose=True):
    if not _sessions:
        if verbose: print("[inference_cnn] no models loaded — returning zeros")
        return np.zeros((len(meta_test), n_classes), dtype=np.float32)

    t0 = time.time()

    if "window_id" not in meta_test.columns:
        meta_test = meta_test.copy()
        meta_test["window_id"] = meta_test.groupby("filename").cumcount()

    groups = meta_test.reset_index().groupby("filename")
    path_by_name = {p.name: p for p in test_paths}

    scores = np.zeros((len(meta_test), n_classes), dtype=np.float32)
    n_files = len(groups)
    t_read = t_mel = t_run = 0.0
    _log_every = max(1, n_files // 10)

    for fi, (fname, g) in enumerate(groups):
        fp = path_by_name.get(fname)
        if fp is None:
            continue

        tr = time.time()
        try:
            wav, _sr = sf.read(str(fp), dtype="float32", always_2d=False)
            if wav.ndim == 2: wav = wav.mean(axis=1)
            need = FILE_SEC * SR
            if len(wav) < need:
                wav = np.pad(wav, (0, need - len(wav)))
            else:
                wav = wav[:need]
        except Exception:
            continue
        t_read += time.time() - tr

        idx_map = np.full(N_WIN, -1, dtype=np.int64)
        for _, r in g.iterrows():
            wi = int(r["window_id"])
            if 0 <= wi < N_WIN:
                idx_map[wi] = int(r["index"])

        acc = np.zeros((N_WIN, n_classes), dtype=np.float32)
        count = 0
        for shift_sec in tta_shifts_sec:
            tm = time.time()
            mel_batch = _file_to_mel_batch(wav, shift_samples=int(shift_sec * SR))
            t_mel += time.time() - tm

            tr2 = time.time()
            for _, sess in _sessions:
                iname = sess.get_inputs()[0].name
                p = sess.run(None, {iname: mel_batch})[0]
                acc += p
                count += 1
            t_run += time.time() - tr2

        acc /= max(1, count)

        for wi in range(N_WIN):
            mi = idx_map[wi]
            if mi >= 0:
                scores[mi] = acc[wi]

        if verbose and (fi + 1) % _log_every == 0:
            el = time.time() - t0
            eta = el * (n_files - fi - 1) / max(1, fi + 1)
            print(f"  CNN [{fi+1}/{n_files}] {el:.0f}s  eta {eta:.0f}s  "
                  f"read={t_read:.1f}  mel={t_mel:.1f}  run={t_run:.1f}")

    total = time.time() - t0
    print(f"[inference_cnn] done. total={total:.1f}s  "
          f"(read={t_read:.1f}s mel={t_mel:.1f}s run={t_run:.1f}s)")
    return scores
