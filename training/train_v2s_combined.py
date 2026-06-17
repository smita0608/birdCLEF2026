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
import torchvision
import librosa
import timm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

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


class SpecAugment(nn.Module):
    def __init__(self, freq_mask_max=10, freq_mask_n=3, freq_p=0.3,
                 time_mask_max=20, time_mask_n=3, time_p=0.3):
        super().__init__()
        self.fm_max = freq_mask_max; self.fm_n = freq_mask_n; self.fp = freq_p
        self.tm_max = time_mask_max; self.tm_n = time_mask_n; self.tp = time_p

    def forward(self, mel):
        if not self.training or self.fm_max <= 0:
            return mel
        B, C, Fd, Td = mel.shape
        for _ in range(self.fm_n):
            if random.random() < self.fp:
                w = random.randint(1, self.fm_max)
                if w < Fd:
                    f0 = random.randint(0, Fd - w)
                    mel[:, :, f0:f0 + w, :] = 0.0
        for _ in range(self.tm_n):
            if random.random() < self.tp:
                w = random.randint(1, self.tm_max)
                if w < Td:
                    t0 = random.randint(0, Td - w)
                    mel[:, :, :, t0:t0 + w] = 0.0
        return mel


class BirdSED(nn.Module):
    def __init__(self, backbone_name, num_classes=234, drop_path_rate=0.15, specaug_enabled=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=1,
            features_only=True, drop_path_rate=drop_path_rate,
        )
        feat_dim = self.backbone.feature_info.channels()[-1]
        self.head = AttHead(feat_dim, num_classes, hidden_chans=1024, p=0.5)
        self.normalize = NormalizeMelSpec()
        self.specaug = SpecAugment() if specaug_enabled else nn.Identity()
        print(f'BirdSED: feat_dim={feat_dim}, head_hidden=1024, specaug={specaug_enabled}')

    def forward(self, mel):
        mel = self.normalize(mel)
        mel = self.specaug(mel)
        feat = self.backbone(mel)[-1]
        return self.head(feat)


class FocalLossBCE(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, bce_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
        self.bce_weight = bce_weight; self.focal_weight = focal_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
    def forward(self, logits, targets):
        focal = torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=logits, targets=targets, alpha=self.alpha, gamma=self.gamma, reduction='mean')
        return self.bce_weight * self.bce(logits, targets) + self.focal_weight * focal


def compute_mel_raw(wav):
    S = librosa.feature.melspectrogram(y=wav, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                         n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
    return librosa.power_to_db(S, top_db=80.0).astype(np.float32)


IMG_T = CLIP_SAMPLES // HOP + 1


def parse_hms_to_seconds(s):
    h, m, sec = s.strip().split(':')
    return int(h) * 3600 + int(m) * 60 + int(sec)


def load_labeled_soundscape_rows(csv_path, train_files):
    df = pd.read_csv(csv_path)
    df = df[df['filename'].isin(set(train_files))].reset_index(drop=True)
    df['start_sec'] = df['start'].apply(parse_hms_to_seconds)
    df['end_sec'] = df['end'].apply(parse_hms_to_seconds)
    df['species_set'] = df['primary_label'].apply(
        lambda s: [x.strip() for x in str(s).split(';') if x.strip()])
    return df


class CombinedDataset(Dataset):
    def __init__(self, focal_df, train_dir, class2idx,
                 pseudo_df=None, pseudo_sampling_prob=0.4,
                 labeled_sound_df=None, labeled_sound_prob=0.2,
                 trim_min_prob=0.1, label_smoothing=0.05, is_train=True):
        self.focal = focal_df.reset_index(drop=True)
        self.train_dir = Path(train_dir)
        self.class2idx = class2idx
        self.is_train = is_train
        self.label_smoothing = label_smoothing
        self.trim_min_prob = trim_min_prob
        self.pseudo_sampling_prob = pseudo_sampling_prob
        self.labeled_sound_prob = labeled_sound_prob
        self.num_classes = len(class2idx)

        self.pseudo_df = pseudo_df
        if pseudo_df is not None and len(pseudo_df) > 0:
            self.pseudo_by_class = {}
            for i, cls in enumerate(pseudo_df['primary_label'].values):
                self.pseudo_by_class.setdefault(cls, []).append(i)
            self.pseudo_classes = set(self.pseudo_by_class.keys())
        else:
            self.pseudo_by_class = {}
            self.pseudo_classes = set()

        self.labeled_sound_df = labeled_sound_df
        self.n_labeled_sound = len(labeled_sound_df) if labeled_sound_df is not None else 0
        if labeled_sound_df is not None:
            self.labeled_sound_valid_idx = [
                i for i in range(len(labeled_sound_df))
                if any(s in class2idx for s in labeled_sound_df.iloc[i]['species_set'])
            ]
        else:
            self.labeled_sound_valid_idx = []

    def __len__(self):
        return len(self.focal)

    def _load_5s(self, path, start_sec=None):
        try:
            with sf.SoundFile(str(path)) as f:
                total = f.frames
                if start_sec is not None:
                    start = int(start_sec * SR)
                    if start + CLIP_SAMPLES > total:
                        start = max(0, total - CLIP_SAMPLES)
                elif self.is_train and total > CLIP_SAMPLES:
                    start = random.randint(0, total - CLIP_SAMPLES)
                else:
                    start = 0
                f.seek(start)
                wav = f.read(CLIP_SAMPLES, dtype='float32', always_2d=False)
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
            if len(wav) < CLIP_SAMPLES:
                wav = np.pad(wav, (0, CLIP_SAMPLES - len(wav)))
            return wav[:CLIP_SAMPLES]
        except Exception:
            return np.zeros(CLIP_SAMPLES, dtype=np.float32)

    def _focal_target(self, row):
        y = np.zeros(self.num_classes, dtype=np.float32)
        if row['primary_label'] in self.class2idx:
            y[self.class2idx[row['primary_label']]] = 1.0
        if 'secondary_labels' in row and isinstance(row['secondary_labels'], str) and row['secondary_labels']:
            for s in row['secondary_labels'].strip("[]").replace("'", "").split(','):
                s = s.strip()
                if s in self.class2idx:
                    y[self.class2idx[s]] = 0.5
        y = np.clip(y, 0.0, 1.0)
        if self.label_smoothing is not None and self.label_smoothing > 0:
            y = y * (1.0 - self.label_smoothing) + self.label_smoothing * y.sum() / self.num_classes
        return y

    def _pseudo_target(self, pseudo_row):
        y = np.array(pseudo_row['labels'], dtype=np.float32).copy()
        y[y < self.trim_min_prob] = 0.0
        return y

    def _labeled_sound_target(self, sound_row):
        y = np.zeros(self.num_classes, dtype=np.float32)
        for s in sound_row['species_set']:
            if s in self.class2idx:
                y[self.class2idx[s]] = 1.0
        if self.label_smoothing is not None and self.label_smoothing > 0:
            y = y * (1.0 - self.label_smoothing) + self.label_smoothing * y.sum() / self.num_classes
        return y

    def __getitem__(self, i):
        if (self.is_train and self.n_labeled_sound > 0 and self.labeled_sound_valid_idx
                and random.random() < self.labeled_sound_prob):
            sound_idx = random.choice(self.labeled_sound_valid_idx)
            sound_row = self.labeled_sound_df.iloc[sound_idx]
            audio_path = BASE / 'train_soundscapes' / sound_row['filename']
            wav = self._load_5s(audio_path, start_sec=int(sound_row['start_sec']))
            target = self._labeled_sound_target(sound_row)
        else:
            row = self.focal.iloc[i]
            focal_cls = row['primary_label']
            use_pseudo = (
                self.is_train
                and self.pseudo_df is not None
                and random.random() < self.pseudo_sampling_prob
                and focal_cls in self.pseudo_classes
            )
            if use_pseudo:
                pseudo_idx = random.choice(self.pseudo_by_class[focal_cls])
                pseudo_row = self.pseudo_df.iloc[pseudo_idx]
                audio_path = BASE / 'train_soundscapes' / pseudo_row['filename']
                wav = self._load_5s(audio_path, start_sec=int(pseudo_row['start_sec']))
                target = self._pseudo_target(pseudo_row)
            else:
                audio_path = self.train_dir / row['filename']
                wav = self._load_5s(audio_path, start_sec=None)
                target = self._focal_target(row)

        mel = compute_mel_raw(wav)
        if mel.shape[-1] < IMG_T:
            mel = np.pad(mel, ((0, 0), (0, IMG_T - mel.shape[-1])))
        else:
            mel = mel[:, :IMG_T]
        return torch.from_numpy(mel[None]), torch.from_numpy(target)


def sydorskyy_mixup(mel_a, y_a, mel_b, y_b):
    return (mel_a + mel_b) * 0.5, torch.clamp(y_a + y_b, 0.0, 1.0)


def train_one_epoch(model, loader, optim, sched, scaler, loss_fn, device, epoch,
                    mixup_p=0.5, use_amp=True):
    model.train()
    t0 = time.time()
    losses = []
    for step, (mel, y) in enumerate(loader):
        mel = mel.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if mixup_p > 0 and random.random() < mixup_p:
            idx = torch.randperm(mel.size(0), device=device)
            mel, y = sydorskyy_mixup(mel, y, mel[idx], y[idx])
        optim.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                out = model(mel)
                loss = loss_fn(out['clipwise_logits'], y)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optim); scaler.update()
        else:
            out = model(mel)
            loss = loss_fn(out['clipwise_logits'], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
        sched.step()
        losses.append(loss.item())
        if step == 0:
            print(f'  e{epoch:02d} first batch took {time.time() - t0:.1f}s')
    return float(np.mean(losses))


@torch.no_grad()
def eval_soundscape_oof(model, meta, device, use_amp=True):
    model.eval()
    preds = np.zeros((len(meta), 234), dtype=np.float32)
    for fname, g in meta.reset_index(drop=True).groupby('filename'):
        wav, _ = sf.read(str(BASE / 'train_soundscapes' / fname), dtype='float32')
        if wav.ndim == 2: wav = wav.mean(axis=1)
        need = 60 * SR
        wav = wav[:need] if len(wav) >= need else np.pad(wav, (0, need - len(wav)))
        mels = np.zeros((12, 1, N_MELS, IMG_T), dtype=np.float32)
        for wi in range(12):
            s = wi * CLIP_SAMPLES
            m = compute_mel_raw(wav[s:s + CLIP_SAMPLES])
            if m.shape[-1] < IMG_T:
                m = np.pad(m, ((0, 0), (0, IMG_T - m.shape[-1])))
            else:
                m = m[:, :IMG_T]
            mels[wi, 0] = m
        x = torch.from_numpy(mels).to(device)
        if use_amp:
            with torch.cuda.amp.autocast():
                probs = model(x)['clipwise_probs'].float().cpu().numpy()
        else:
            probs = model(x)['clipwise_probs'].float().cpu().numpy()
        for _, row in g.iterrows():
            preds[row.name] = probs[int(row['window_id'])]
    return score_predictions(preds, meta), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--backbone', default='tf_efficientnetv2_s.in21k_ft_in1k')
    ap.add_argument('--pretrained', default=str(CKPT_DEFAULT))
    ap.add_argument('--out_dir', default=str(WORK / 'sed_v2s_e14_out'))
    ap.add_argument('--pseudo_parquet', default=str(WORK / 'pseudo_labels_sydorskyy.parquet'))
    ap.add_argument('--labeled_sound_csv', default=str(BASE / 'train_soundscapes_labels.csv'))
    ap.add_argument('--soundscape_split_json', default=str(WORK / 'labeled_soundscape_split.json'))
    ap.add_argument('--pseudo_sampling_prob', type=float, default=0.4)
    ap.add_argument('--labeled_sound_prob', type=float, default=0.2)
    ap.add_argument('--pseudo_trim_min_prob', type=float, default=0.1)
    ap.add_argument('--pseudo_primary_min_prob', type=float, default=0.5)
    ap.add_argument('--label_smoothing', type=float, default=0.05)
    ap.add_argument('--mixup_p', type=float, default=0.5)
    ap.add_argument('--use_balanced_sampler', action='store_true')
    ap.add_argument('--balance_power', type=float, default=0.5,
                    help='Sampling weight = count^-balance_power (0.5 = sqrt-inverse)')
    ap.add_argument('--no_pseudo', action='store_true')
    ap.add_argument('--no_labeled_sound', action='store_true')
    ap.add_argument('--no_specaug', action='store_true')
    ap.add_argument('--no_amp', action='store_true')
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--max_focal_samples', type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.epochs = 3
        args.max_focal_samples = 50
        print('SMOKE MODE: 3 epochs, 50 focal/class')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}  LR: {args.lr}  mixup_p: {args.mixup_p}  label_smoothing: {args.label_smoothing}')
    print(f'Balanced sampler: {args.use_balanced_sampler}  Labeled sound prob: {args.labeled_sound_prob}')

    classes = pd.read_csv(BASE / 'sample_submission.csv', nrows=1).columns[1:].tolist()
    class2idx = {c: i for i, c in enumerate(classes)}

    split = json.loads(Path(args.soundscape_split_json).read_text())
    train_sound_files = split['train_files']
    val_sound_files = split['val_files']
    print(f'Soundscape split: train={len(train_sound_files)} files, val={len(val_sound_files)} files')

    print(f'Building model: {args.backbone}')
    model = BirdSED(args.backbone, num_classes=234, drop_path_rate=0.15,
                    specaug_enabled=(not args.no_specaug)).to(device)
    ckpt = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    bb_state = {f'backbone.{k}': v for k, v in ckpt.items()}
    missing, unexpected = model.load_state_dict(bb_state, strict=False)
    head_missing = [k for k in missing if k.startswith('head.')]
    backbone_missing = [k for k in missing if k.startswith('backbone.')]
    print(f'Pretrained load: missing total={len(missing)}, head_missing={len(head_missing)}, backbone_missing={len(backbone_missing)}')

    train_csv = pd.read_csv(BASE / 'train.csv')
    train_csv = train_csv[train_csv['primary_label'].isin(class2idx)].reset_index(drop=True)
    if args.max_focal_samples > 0:
        train_csv = train_csv.groupby('primary_label').head(args.max_focal_samples).reset_index(drop=True)
    print(f'Focal rows: {len(train_csv)}')

    pseudo_df = None
    if not args.no_pseudo:
        pseudo_df = pd.read_parquet(args.pseudo_parquet)
        pseudo_df = pseudo_df[pseudo_df['primary_label_prob'] > args.pseudo_primary_min_prob].reset_index(drop=True)
        pseudo_df = pseudo_df[pseudo_df['primary_label'].isin(class2idx)].reset_index(drop=True)
        n_classes_with_pseudo = pseudo_df['primary_label'].nunique()
        print(f'Pseudo rows: {len(pseudo_df)} ({n_classes_with_pseudo}/{len(classes)} classes)')

    labeled_sound_df = None
    if not args.no_labeled_sound:
        labeled_sound_df = load_labeled_soundscape_rows(args.labeled_sound_csv, train_sound_files)
        print(f'Labeled soundscape rows: {len(labeled_sound_df)} (from {len(train_sound_files)} train files)')

    ds = CombinedDataset(
        focal_df=train_csv, train_dir=BASE / 'train_audio', class2idx=class2idx,
        pseudo_df=pseudo_df, pseudo_sampling_prob=args.pseudo_sampling_prob,
        labeled_sound_df=labeled_sound_df, labeled_sound_prob=args.labeled_sound_prob,
        trim_min_prob=args.pseudo_trim_min_prob, label_smoothing=args.label_smoothing,
        is_train=True,
    )

    if args.use_balanced_sampler:
        class_counts = train_csv['primary_label'].value_counts().to_dict()
        sample_weights = np.array([
            1.0 / (class_counts.get(c, 1) ** args.balance_power)
            for c in train_csv['primary_label'].values
        ], dtype=np.float64)
        sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(ds), replacement=True)
        dl = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
        print(f'Balanced sampler ON (power={args.balance_power}); weights min={sample_weights.min():.3f} max={sample_weights.max():.3f}')
    else:
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    print(f'Steps per epoch: {len(dl)}  total_steps: {len(dl) * args.epochs}')

    full_meta = build_soundscape_val_meta(BASE, classes=classes,
                                            only_fully_labeled=True, n_windows_per_file=12)
    val_meta = full_meta[full_meta['filename'].isin(set(val_sound_files))].reset_index(drop=True)
    print(f'OOF val meta: {len(val_meta)} rows across {val_meta["filename"].nunique()} held-out files')
    assert len(val_meta) > 0, 'No held-out OOF rows; check split'

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8, betas=(0.9, 0.999))
    total_steps = len(dl) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optim, T_0=total_steps, T_mult=1, eta_min=1e-6, last_epoch=-1)
    use_amp = not args.no_amp
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    loss_fn = FocalLossBCE(alpha=0.25, gamma=2.0, bce_weight=1.0, focal_weight=1.0)
    print(f'Loss: FocalLossBCE  Schedule: CosineAnnealingWarmRestarts T_0={total_steps}  AMP: {use_amp}')

    history = []
    best_oof = -1.0
    best_state = None
    best_preds = None
    best_epoch = -1
    for ep in range(args.epochs):
        tr_loss = train_one_epoch(model, dl, optim, sched, scaler, loss_fn, device, ep,
                                   mixup_p=args.mixup_p, use_amp=use_amp)
        oof_auc, preds = eval_soundscape_oof(model, val_meta, device, use_amp=use_amp)
        is_best = oof_auc > best_oof
        marker = '  ✓ NEW BEST' if is_best else ''
        print(f'  epoch {ep:02d}  loss={tr_loss:.4f}  heldout_OOF_AUC={oof_auc:.5f}{marker}')
        history.append({'epoch': ep, 'train_loss': tr_loss, 'heldout_oof_auc': oof_auc, 'is_best': is_best})
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
        print(f'\n  best epoch: {best_epoch}  best heldout OOF: {best_oof:.5f}')
    (out_dir / 'history.json').write_text(json.dumps(history, indent=2))
    final = max(h['heldout_oof_auc'] for h in history)
    print(f'\nFinal heldout OOF (best across all epochs): {final:.5f}')


if __name__ == '__main__':
    main()
