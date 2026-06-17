from __future__ import annotations

import argparse
import json
import random
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
import timm
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/home/user/work/tf-mp/kaggle/birdclef_upgrade')
from soundscape_val import build_soundscape_val_meta, score_predictions

BASE = Path('/home/user/work/tf-mp/kaggle/birdclef-2026')
WORK = Path('/home/user/work/tf-mp/kaggle/working_949')
CKPT_DEFAULT = Path('/home/user/work/tf-mp/kaggle/birdclef-2026/bird-clef-2025-all-pretrained-models/models_2025/tf_efficientnetv2_s_in21k_pretrain_from_bigXCV2Ext_swa.ckpt')

SR = 32_000
CLIP_SEC = 5.0
CLIP_SAMPLES = int(SR * CLIP_SEC)
N_FFT = 2048
HOP = 512
N_MELS = 128
FMIN, FMAX = 20, 16_000


class GeMFreq(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p, dtype=torch.float32))
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            kernel_size=(x.size(-2), 1)).pow(1.0 / self.p)


class AttHead(nn.Module):
    def __init__(self, in_chans, num_class=234, hidden_chans=1024, p=0.5):
        super().__init__()
        self.pooling = GeMFreq()
        self.dense_layers = nn.Sequential(
            nn.Dropout(p / 2),
            nn.Linear(in_chans, hidden_chans),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.attention = nn.Conv1d(hidden_chans, num_class, kernel_size=1, bias=True)
        self.fix_scale = nn.Conv1d(hidden_chans, num_class, kernel_size=1, bias=True)

    def forward(self, feat):
        feat = self.pooling(feat).squeeze(-2).permute(0, 2, 1)
        feat = self.dense_layers(feat).permute(0, 2, 1)

        time_att = torch.tanh(self.attention(feat))
        feat_v = self.fix_scale(feat)

        att_w = torch.softmax(time_att, dim=-1)
        clipwise_logits = torch.sum(feat_v * att_w, dim=-1)
        clipwise_probs = torch.sum(torch.sigmoid(feat_v) * att_w, dim=-1)

        framewise_logits_max = feat_v.max(dim=-1).values
        return {
            'clipwise_logits': clipwise_logits,
            'clipwise_probs': clipwise_probs,
            'framewise_logits_max': framewise_logits_max,
        }


class NormalizeMelSpec(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, X):
        mean = X.mean((-2, -1), keepdim=True)
        std = X.std((-2, -1), keepdim=True)
        Xstd = (X - mean) / (std + self.eps)
        Xmax = torch.amax(Xstd, dim=(-2, -1), keepdim=True)
        Xmin = torch.amin(Xstd, dim=(-2, -1), keepdim=True)
        return (Xstd - Xmin) / (Xmax - Xmin + self.eps)


class BirdSED(nn.Module):
    def __init__(self, backbone_name, num_classes=234, drop_path_rate=0.15):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=1,
            features_only=True,
            drop_path_rate=drop_path_rate,
        )
        feat_dim = self.backbone.feature_info.channels()[-1]
        self.head = AttHead(feat_dim, num_classes, hidden_chans=1024, p=0.5)
        self.normalize = NormalizeMelSpec()
        print(f'BirdSED: backbone last-stage feat_dim={feat_dim}, head hidden=1024')

    def forward(self, mel):
        mel = self.normalize(mel)
        feat = self.backbone(mel)[-1]
        return self.head(feat)


def focal_bce_with_logits(logits, targets, gamma=2.0, label_smoothing=0.005):
    targets_smooth = targets * (1 - label_smoothing) + label_smoothing * 0.5
    bce = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none')
    p = torch.sigmoid(logits)
    p_t = torch.where(targets > 0.5, p, 1 - p)
    focal = (1 - p_t) ** gamma * bce
    return focal.mean()


def compute_mel_raw(wav):
    S = librosa.feature.melspectrogram(y=wav, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                         n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
    M = librosa.power_to_db(S, top_db=80.0)
    return M.astype(np.float32)


IMG_T = CLIP_SAMPLES // HOP + 1


class FocalDataset(Dataset):
    def __init__(self, df, train_dir, class2idx, is_train=True):
        self.df = df.reset_index(drop=True)
        self.train_dir = Path(train_dir)
        self.class2idx = class2idx
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def _load_5s(self, path):
        try:
            with sf.SoundFile(str(path)) as f:
                total = f.frames
                if total >= CLIP_SAMPLES:
                    start = random.randint(0, total - CLIP_SAMPLES) if self.is_train else 0
                    f.seek(start)
                    wav = f.read(CLIP_SAMPLES, dtype='float32', always_2d=False)
                else:
                    wav = f.read(dtype='float32', always_2d=False)
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
            if len(wav) < CLIP_SAMPLES:
                wav = np.pad(wav, (0, CLIP_SAMPLES - len(wav)))
            return wav[:CLIP_SAMPLES]
        except Exception:
            return np.zeros(CLIP_SAMPLES, dtype=np.float32)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        wav = self._load_5s(self.train_dir / row['filename'])
        mel = compute_mel_raw(wav)
        if mel.shape[-1] < IMG_T:
            mel = np.pad(mel, ((0, 0), (0, IMG_T - mel.shape[-1])))
        else:
            mel = mel[:, :IMG_T]

        y = np.zeros(len(self.class2idx), dtype=np.float32)
        if row['primary_label'] in self.class2idx:
            y[self.class2idx[row['primary_label']]] = 1.0
        if 'secondary_labels' in row and isinstance(row['secondary_labels'], str) and row['secondary_labels']:
            for s in row['secondary_labels'].strip("[]").replace("'", "").split(','):
                s = s.strip()
                if s in self.class2idx:
                    y[self.class2idx[s]] = 0.5

        return torch.from_numpy(mel[None]), torch.from_numpy(y)


def mixup_audio(mel_a, y_a, mel_b, y_b, lam=0.5):
    return lam * mel_a + (1 - lam) * mel_b, lam * y_a + (1 - lam) * y_b


def train_one_epoch(model, loader, optim, sched, scaler, device, epoch, mixup_p=0.5):
    model.train()
    t0 = time.time()
    losses = []
    for step, (mel, y) in enumerate(loader):
        mel = mel.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if mixup_p > 0 and random.random() < mixup_p:
            idx = torch.randperm(mel.size(0), device=device)
            mel, y = mixup_audio(mel, y, mel[idx], y[idx], lam=0.5)

        optim.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(mel)
            loss = 0.5 * F.binary_cross_entropy_with_logits(out['clipwise_logits'], y) + \
                   0.5 * F.binary_cross_entropy_with_logits(out['framewise_logits_max'], y)

        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optim)
        scaler.update()
        sched.step()

        losses.append(loss.item())
        if step == 0:
            print(f'  e{epoch:02d} first batch took {time.time() - t0:.1f}s')
    return float(np.mean(losses))


@torch.no_grad()
def eval_soundscape_oof(model, meta, device):
    model.eval()
    preds = np.zeros((len(meta), 234), dtype=np.float32)
    groups = meta.reset_index(drop=True).groupby('filename')
    for fname, g in groups:
        wav, _ = sf.read(str(BASE / 'train_soundscapes' / fname), dtype='float32')
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        need = 60 * SR
        wav = wav[:need] if len(wav) >= need else np.pad(wav, (0, need - len(wav)))
        mels = np.zeros((12, 1, N_MELS, IMG_T), dtype=np.float32)
        for wi in range(12):
            s = wi * CLIP_SAMPLES
            chunk = wav[s:s + CLIP_SAMPLES]
            m = compute_mel_raw(chunk)
            if m.shape[-1] < IMG_T:
                m = np.pad(m, ((0, 0), (0, IMG_T - m.shape[-1])))
            else:
                m = m[:, :IMG_T]
            mels[wi, 0] = m
        x = torch.from_numpy(mels).to(device)
        with torch.cuda.amp.autocast():
            out = model(x)
            probs = out['clipwise_probs'].float().cpu().numpy()
        for _, row in g.iterrows():
            preds[row.name] = probs[int(row['window_id'])]
    return score_predictions(preds, meta), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--backbone', default='tf_efficientnetv2_s.in21k_ft_in1k')
    ap.add_argument('--pretrained', default=str(CKPT_DEFAULT))
    ap.add_argument('--out_dir', default=str(WORK / 'sed_v2s_full_out'))
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--max_focal_samples', type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.epochs = 3
        args.max_focal_samples = 50
        print('SMOKE MODE: 3 epochs, batch 32, 50 samples/class max')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}  LR: {args.lr}')

    classes = pd.read_csv(BASE / 'sample_submission.csv', nrows=1).columns[1:].tolist()
    class2idx = {c: i for i, c in enumerate(classes)}
    print(f'Classes: {len(classes)}')

    print(f'Building model: {args.backbone}  drop_path=0.15  hidden_chans=1024')
    model = BirdSED(args.backbone, num_classes=234, drop_path_rate=0.15).to(device)

    ckpt = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    bb_state = {f'backbone.{k}': v for k, v in ckpt.items()}
    missing, unexpected = model.load_state_dict(bb_state, strict=False)
    head_missing = [k for k in missing if k.startswith('head.')]
    backbone_missing = [k for k in missing if not k.startswith('head.')]
    print(f'Pretrained load: {len(missing)} missing total')
    print(f'  Head missing (expected — new head): {len(head_missing)}')
    print(f'  Backbone missing (should be ~0 with features_only): {len(backbone_missing)}')
    for k in backbone_missing[:5]:
        print(f'    {k}')

    train_csv = pd.read_csv(BASE / 'train.csv')
    train_csv = train_csv[train_csv['primary_label'].isin(class2idx)].reset_index(drop=True)
    if args.max_focal_samples > 0:
        train_csv = train_csv.groupby('primary_label').head(args.max_focal_samples).reset_index(drop=True)
    print(f'Train rows: {len(train_csv)}')

    ds = FocalDataset(train_csv, BASE / 'train_audio', class2idx, is_train=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                     num_workers=args.num_workers, pin_memory=True, drop_last=True)
    print(f'Steps per epoch: {len(dl)}')

    meta = build_soundscape_val_meta(BASE, classes=classes,
                                      only_fully_labeled=True, n_windows_per_file=12)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-8)
    total_steps = len(dl) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, total_steps=total_steps, pct_start=0.1, anneal_strategy='cos',
    )
    scaler = torch.cuda.amp.GradScaler()

    history = []
    best_oof = -1.0
    best_state = None
    best_preds = None
    best_epoch = -1
    for ep in range(args.epochs):
        tr_loss = train_one_epoch(model, dl, optim, sched, scaler, device, ep, mixup_p=0.3)
        oof_auc, preds = eval_soundscape_oof(model, meta, device)
        is_best = oof_auc > best_oof
        marker = '  ✓ NEW BEST' if is_best else ''
        print(f'  epoch {ep:02d}  loss={tr_loss:.4f}  soundscape_OOF_AUC={oof_auc:.5f}{marker}')
        history.append({'epoch': ep, 'train_loss': tr_loss, 'soundscape_oof_auc': oof_auc, 'is_best': is_best})
        if is_best:
            best_oof = oof_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_preds = preds.copy()
            best_epoch = ep
            torch.save(best_state, out_dir / 'model_best.pt')
            np.save(out_dir / 'oof_preds_best.npy', best_preds)

    torch.save(model.state_dict(), out_dir / 'model_final.pt')
    if best_state is not None:
        torch.save(best_state, out_dir / 'model_best.pt')
        np.save(out_dir / 'oof_preds_best.npy', best_preds)
        print(f'\n  best epoch: {best_epoch}  best OOF: {best_oof:.5f}')
    (out_dir / 'history.json').write_text(json.dumps(history, indent=2))
    if best_state is not None:
        torch.save(best_state, out_dir / 'model.pt')
        np.save(out_dir / 'oof_preds.npy', best_preds)
    final = max(h['soundscape_oof_auc'] for h in history)
    print(f'\nFinal OOF (best across all epochs): {final:.5f}')
    if final >= 0.85:
        print('  ✓ STRONG — proceed to full training')
    elif final >= 0.75:
        print('  ~ PROMISING — full training should push higher')
    elif final >= 0.65:
        print('  ~ MARGINAL — about the prior naive V2-s baseline. Architecture not fully bridged.')
    else:
        print('  ✗ WEAK — something is still wrong')


if __name__ == '__main__':
    main()
