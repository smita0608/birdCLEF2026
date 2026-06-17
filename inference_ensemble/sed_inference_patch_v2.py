
import sys, time
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from pathlib import Path

_SED_V2_SUCCESS = False

try:
    import onnxruntime as ort
except ImportError:
    print("[SED v2] onnxruntime not available — skipping")
    raise

try:
    def _find_dir(candidates):
        for p in candidates:
            if Path(p).exists() and list(Path(p).glob("*.onnx")):
                return p
        return None

    focal_dir = _find_dir([
        "/kaggle/input/birdclef2026-sed-b0-jft",
        "/kaggle/input/datasets/smitapriyadarshani/birdclef2026-sed-b0-jft",
    ])
    snd_dir = _find_dir([
        "/kaggle/input/birdclef2026-sed-soundscape",
        "/kaggle/input/datasets/smitapriyadarshani/birdclef2026-sed-soundscape",
    ])

    focal_paths = []
    if focal_dir:
        focal_paths = sorted([p for p in Path(focal_dir).glob("sed_b0_jft_fold*.onnx")
                              if any(f"fold{i}." in p.name for i in [0, 1])])
    snd_paths = sorted(Path(snd_dir).glob("sed_b0_jft_soundscape_fold*.onnx")) if snd_dir else []

    print(f"[SED v2] focal models: {len(focal_paths)} from {focal_dir}")
    for p in focal_paths: print(f"    {p.name}")
    print(f"[SED v2] soundscape models: {len(snd_paths)} from {snd_dir}")
    for p in snd_paths: print(f"    {p.name}")
    assert focal_paths or snd_paths, "No SED ONNX files found in any expected location"

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 4
    focal_sessions = [ort.InferenceSession(str(p), sess_options=so,
                                           providers=["CPUExecutionProvider"])
                      for p in focal_paths]
    snd_sessions = [ort.InferenceSession(str(p), sess_options=so,
                                         providers=["CPUExecutionProvider"])
                    for p in snd_paths]

    SR = 32000
    CLIP_SEC = 5.0
    WIN_SAMPLES = int(SR * CLIP_SEC)
    FILE_SEC = 60
    N_WIN = 12
    N_FFT, HOP, N_MELS = 2048, 512, 128
    FMIN, FMAX = 50, 14000

    _mel_t = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        f_min=FMIN, f_max=FMAX, power=2.0)
    _db_t = torchaudio.transforms.AmplitudeToDB(top_db=80)

    _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def _file_to_mel_3ch_batch(wav60):
        chunks = np.zeros((N_WIN, WIN_SAMPLES), dtype=np.float32)
        for wi in range(N_WIN):
            s = wi * WIN_SAMPLES
            chunks[wi] = wav60[s:s + WIN_SAMPLES]
        x = torch.from_numpy(chunks)
        with torch.no_grad():
            m = _db_t(_mel_t(x))
            m = (m - m.mean(dim=(-2, -1), keepdim=True)) / \
                (m.std(dim=(-2, -1), keepdim=True) + 1e-6)
        m = m.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
        m = (m - _IMAGENET_MEAN) / _IMAGENET_STD
        return m.numpy().astype(np.float32)

    print(f"[SED v2] Running inference on {len(test_paths)} test files...")
    sed_t0 = time.time()
    focal_scores = np.zeros((len(meta_te), N_CLASSES), dtype=np.float32)
    snd_scores   = np.zeros((len(meta_te), N_CLASSES), dtype=np.float32)

    if "window_id" not in meta_te.columns:
        meta_te = meta_te.copy()
        meta_te["window_id"] = meta_te.groupby("filename").cumcount()
    groups = meta_te.reset_index().groupby("filename")
    path_by_name = {p.name: p for p in test_paths}

    for fname, g in groups:
        fp = path_by_name.get(fname)
        if fp is None: continue
        try:
            wav, _ = sf.read(str(fp), dtype="float32", always_2d=False)
            if wav.ndim == 2: wav = wav.mean(axis=1)
            need = FILE_SEC * SR
            if len(wav) < need: wav = np.pad(wav, (0, need - len(wav)))
            else: wav = wav[:need]
        except Exception:
            continue

        mel_batch = _file_to_mel_3ch_batch(wav)

        focal_avg = None
        if focal_sessions:
            preds = []
            for sess in focal_sessions:
                iname = sess.get_inputs()[0].name
                preds.append(sess.run(None, {iname: mel_batch})[0])
            focal_avg = np.mean(preds, axis=0)

        snd_avg = None
        if snd_sessions:
            preds = []
            for sess in snd_sessions:
                iname = sess.get_inputs()[0].name
                preds.append(sess.run(None, {iname: mel_batch})[0])
            snd_avg = np.mean(preds, axis=0)

        for _, row in g.iterrows():
            wi = int(row["window_id"])
            mi = int(row["index"])
            if 0 <= wi < N_WIN:
                if focal_avg is not None: focal_scores[mi] = focal_avg[wi]
                if snd_avg   is not None: snd_scores[mi]   = snd_avg[wi]

    print(f"[SED v2] inference took {(time.time()-sed_t0)/60:.1f} min")

    FOCAL_SED_WEIGHT = 0.40
    SND_SED_WEIGHT   = 0.60

    if focal_sessions and snd_sessions:
        sed_scores = (FOCAL_SED_WEIGHT * focal_scores +
                      SND_SED_WEIGHT   * snd_scores).astype(np.float32)
        print(f"[SED v2] SED ensemble = focal({FOCAL_SED_WEIGHT}) + soundscape({SND_SED_WEIGHT})")
    elif focal_sessions:
        sed_scores = focal_scores
        print("[SED v2] focal-only (soundscape SED dataset not attached)")
    else:
        sed_scores = snd_scores
        print("[SED v2] soundscape-only (focal SED dataset not attached)")

    submission_path = Path("submission.csv")
    assert submission_path.exists(), "submission.csv not found — baseline failed?"
    baseline_sub = pd.read_csv(submission_path)
    baseline_probs = baseline_sub[PRIMARY_LABELS].to_numpy().astype(np.float32)
    assert baseline_probs.shape == sed_scores.shape

    def _rank_mean(x):
        r = np.zeros_like(x, dtype=np.float32)
        n = x.shape[0]
        for c in range(x.shape[1]):
            r[:, c] = (np.argsort(np.argsort(x[:, c])) / max(n - 1, 1)).astype(np.float32)
        return r

    def _quantile_mix(x, alpha=0.5):
        return alpha * x + (1 - alpha) * _rank_mean(x)

    SED_WEIGHT  = 0.35 if (focal_sessions and snd_sessions) else 0.30
    BASE_WEIGHT = 1.0 - SED_WEIGHT

    blended = (
        BASE_WEIGHT * _quantile_mix(baseline_probs, alpha=0.5) +
        SED_WEIGHT  * _quantile_mix(sed_scores,    alpha=0.5)
    )
    blended = np.clip(blended, 0.0, 1.0).astype(np.float32)

    new_sub = pd.DataFrame(blended, columns=PRIMARY_LABELS)
    new_sub.insert(0, "row_id", baseline_sub["row_id"].values)
    new_sub.to_csv("submission.csv", index=False)

    _SED_V2_SUCCESS = True
    print(f"[SED v2] OK — baseline {BASE_WEIGHT:.0%} + SED {SED_WEIGHT:.0%}")
    print(f"[SED v2] score range: [{blended.min():.4f}, {blended.max():.4f}]  mean: {blended.mean():.4f}")

except Exception as _e:
    print(f"[SED v2] blend SKIPPED: {_e}")
    import traceback; traceback.print_exc()
    print("[SED v2] baseline submission.csv on disk is preserved.")

print(f"[SED v2] _SED_V2_SUCCESS = {_SED_V2_SUCCESS}")
