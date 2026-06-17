from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
sys.modules.setdefault('numpy._core', np.core)
sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
sys.modules.setdefault('numpy._core._multiarray_umath', np.core._multiarray_umath)

import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from soundscape_val import build_soundscape_val_meta, score_predictions, split_by_site

SR = 32_000
CLIP_SEC = 5.0
WIN_SAMPLES = int(SR * CLIP_SEC)
FILE_SEC = 60
N_WINDOWS = 12
N_FFT = 2048
HOP = 512
N_MELS = 128
FMIN, FMAX = 50, 14_000


def compute_mel(wav: np.ndarray) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=wav, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX, power=2.0,
    )
    M = librosa.power_to_db(S, top_db=80.0)
    M = (M - M.mean()) / (M.std() + 1e-6)
    return M.astype(np.float32)


def slice_windows(wav60: np.ndarray) -> np.ndarray:
    total = len(wav60)
    chunks = np.zeros((N_WINDOWS, WIN_SAMPLES), dtype=np.float32)
    for wi in range(N_WINDOWS):
        s = wi * WIN_SAMPLES
        s = max(0, min(total - WIN_SAMPLES, s))
        chunks[wi] = wav60[s:s + WIN_SAMPLES]
    return chunks


class AlexanterkapaiV2s(nn.Module):
    def __init__(self, num_classes: int = 234, dropout: float = 0.1):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s.in21k_ft_in1k',
            pretrained=False, num_classes=0, global_pool='avg',
        )
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


class BaiyubyV2sSED(nn.Module):
    def __init__(self, num_classes: int = 234):
        super().__init__()
        import timm
        self.bb = timm.create_model(
            'tf_efficientnetv2_s',
            pretrained=False, in_chans=1, features_only=False, num_classes=0,
            global_pool='',
        )
        feat_c = 1280
        att_dim = 640
        self.att = nn.Sequential(
            nn.Linear(feat_c, att_dim),
            nn.Tanh(),
            nn.Linear(att_dim, num_classes),
        )
        self.fc_att = nn.Linear(feat_c, num_classes)
        self.fc_max = nn.Linear(feat_c, num_classes)

    def forward(self, x):
        feat = self.bb.forward_features(x)
        feat = feat.mean(dim=2)
        feat = feat.transpose(1, 2)

        att_logits = self.att(feat)
        att_w = F.softmax(att_logits, dim=1)

        frame_logits = self.fc_att(feat)
        clip_logits = (att_w * frame_logits).sum(dim=1)

        framemax_logits = self.fc_max(feat).max(dim=1).values

        return 0.5 * (clip_logits + framemax_logits)


def load_alexanterkapai(path: Path, device='cpu') -> nn.Module:
    sd = torch.load(path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    model = AlexanterkapaiV2s(num_classes=234, dropout=0.1)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  loaded alexanterkapai: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"    first missing: {missing[:5]}")
    if unexpected:
        print(f"    first unexpected: {unexpected[:5]}")
    model.eval()
    return model


def load_baiyuby(path: Path, device='cpu') -> nn.Module:
    sd = torch.load(path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and 'model' in sd and isinstance(sd['model'], dict):
        sd = sd['model']
    elif isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    model = BaiyubyV2sSED(num_classes=234)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  loaded baiyuby {path.name}: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"    first missing: {missing[:5]}")
    if unexpected:
        print(f"    first unexpected: {unexpected[:5]}")
    model.eval()
    return model


def run_inference(meta: pd.DataFrame, model: nn.Module, base_dir: Path,
                  in_channels: int, batch_size: int = 12, verbose: bool = True) -> np.ndarray:
    n_classes = 234
    out = np.zeros((len(meta), n_classes), dtype=np.float32)
    meta = meta.reset_index(drop=True)
    groups = meta.groupby('filename')
    n_files = len(groups)
    t0 = time.time()
    t_read = t_mel = t_run = 0.0

    for fi, (fname, g) in enumerate(groups):
        fp = base_dir / 'train_soundscapes' / fname
        tr = time.time()
        try:
            wav, _ = sf.read(str(fp), dtype='float32', always_2d=False)
        except Exception as e:
            if verbose:
                print(f"  skipped {fname}: {e}")
            continue
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        need = FILE_SEC * SR
        if len(wav) < need:
            wav = np.pad(wav, (0, need - len(wav)))
        else:
            wav = wav[:need]
        t_read += time.time() - tr

        tm = time.time()
        chunks = slice_windows(wav)
        mels = np.stack([compute_mel(c) for c in chunks])
        if in_channels == 3:
            mel_batch = np.repeat(mels[:, None, :, :], 3, axis=1)
        else:
            mel_batch = mels[:, None, :, :]
        mel_batch = torch.from_numpy(mel_batch.astype(np.float32))
        t_mel += time.time() - tm

        tr2 = time.time()
        with torch.no_grad():
            logits = model(mel_batch).cpu().numpy()
        t_run += time.time() - tr2

        for _, row in g.iterrows():
            wi = int(row['window_id'])
            out[row.name] = logits[wi]

        if verbose and (fi + 1) % 10 == 0:
            el = time.time() - t0
            eta = el * (n_files - fi - 1) / max(1, fi + 1)
            print(f"  [{fi+1}/{n_files}] {el:.0f}s eta {eta:.0f}s  "
                  f"read={t_read:.1f} mel={t_mel:.1f} run={t_run:.1f}")

    total = time.time() - t0
    print(f"  done in {total:.0f}s  (read={t_read:.1f}s mel={t_mel:.1f}s run={t_run:.1f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='/home/user/work/tf-mp/kaggle/birdclef-2026')
    ap.add_argument('--models', nargs='+', default=['alexanterkapai', 'baiyuby_fold0'])
    ap.add_argument('--save-preds', default='/home/user/work/tf-mp/kaggle/oof_preds')
    args = ap.parse_args()

    base = Path(args.base)
    save_dir = Path(args.save_preds)
    save_dir.mkdir(exist_ok=True, parents=True)

    classes = pd.read_csv(base / 'sample_submission.csv', nrows=1).columns[1:].tolist()
    print(f"Loaded {len(classes)} classes from sample_submission")

    meta = build_soundscape_val_meta(base, classes=classes, only_fully_labeled=True,
                                      n_windows_per_file=N_WINDOWS)
    n_active = int((np.stack(meta['labels'].to_numpy()).sum(0) > 0).sum())
    print(f"Built meta: {len(meta)} windows × {meta['filename'].nunique()} files, "
          f"{n_active} active classes")

    results = {}
    if 'alexanterkapai' in args.models:
        print("\n=== alexanterkapai V2-s (3ch, simple head) ===")
        m = load_alexanterkapai(
            Path('/home/user/work/tf-mp/kaggle/alexanterkapaibirdclef-2026-models/model_fold0.pth'))
        preds_logits = run_inference(meta, m, base, in_channels=3)
        preds = torch.sigmoid(torch.from_numpy(preds_logits)).numpy()
        np.save(save_dir / 'alexanterkapai_v2s.npy', preds)
        auc = score_predictions(preds, meta)
        print(f"  ===> alexanterkapai OOF macro-AUC = {auc:.5f}")
        results['alexanterkapai'] = float(auc)
        for site, sub in split_by_site(meta).items():
            idx = meta.index.isin(sub.index)
            auc_s = score_predictions(preds[idx], sub)
            print(f"      site={site}  ({len(sub)} windows) AUC={auc_s:.5f}")

    if 'baiyuby_fold0' in args.models:
        print("\n=== baiyuby V2-s fold0 (1ch, SED dual-head) ===")
        m = load_baiyuby(
            Path('/home/user/work/tf-mp/kaggle/baiyubybirdclef2026-distill-models/fold0_best.pth'))
        preds_logits = run_inference(meta, m, base, in_channels=1)
        preds = torch.sigmoid(torch.from_numpy(preds_logits)).numpy()
        np.save(save_dir / 'baiyuby_v2s_fold0.npy', preds)
        auc = score_predictions(preds, meta)
        print(f"  ===> baiyuby fold0 OOF macro-AUC = {auc:.5f}")
        results['baiyuby_fold0'] = float(auc)

    with open(save_dir / 'oof_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {save_dir / 'oof_results.json'}")


if __name__ == '__main__':
    main()
