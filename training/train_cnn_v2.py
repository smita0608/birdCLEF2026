
import os, gc, json, math, random, re, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import timm
import torchaudio

warnings.filterwarnings("ignore")

CFG = {
    "COMP_DIR":    "/kaggle/input/birdclef-2026",
    "PSEUDO_DIR":  "/kaggle/input/birdclef2026-pseudo-from-v1",
    "OUT_DIR":     "/kaggle/working",

    "BACKBONE":    "tf_efficientnet_b0_ns",

    "USE_SED":     False,
    "SR":          32000,
    "CLIP_SEC":    5.0,
    "N_FFT":       2048,
    "HOP":         512,
    "N_MELS":      128,
    "FMIN":        50,
    "FMAX":        14000,
    "IMG_SIZE":    (128, 313),

    "EPOCHS":      25,
    "LR":          8e-4,
    "WD":          1e-4,
    "MIXUP_ALPHA": 0.5,
    "LABEL_SMOOTH":0.0,
    "FOLDS_TO_RUN":[0, 1, 2],
    "N_FOLDS":     5,
    "SEED":        42,
    "NUM_WORKERS": 4,
    "MAX_SAMPLES_PER_CLASS": 500,
    "MIN_SAMPLES_PER_CLASS": 5,

    "PSEUDO_BATCH_RATIO": 0.5,

    "EXTERNAL_NOISE_DIR": None,
}

_AUTO_BS = {
    "tf_efficientnet_b0_ns": 32,
    "eca_nfnet_l0":          24,
    "tf_efficientnetv2_s":   24,
}
CFG["BATCH_SIZE"] = _AUTO_BS.get(CFG["BACKBONE"], 24)

os.makedirs(CFG["OUT_DIR"], exist_ok=True)
random.seed(CFG["SEED"]); np.random.seed(CFG["SEED"]); torch.manual_seed(CFG["SEED"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPUS = torch.cuda.device_count() if DEVICE == "cuda" else 0
print("Device:", DEVICE, "N_GPUs:", N_GPUS,
      "Backbone:", CFG["BACKBONE"], "Batch:", CFG["BATCH_SIZE"])

BASE = Path(CFG["COMP_DIR"])
train_csv = pd.read_csv(BASE / "train.csv")
sample_sub = pd.read_csv(BASE / "sample_submission.csv")
CLASSES = sample_sub.columns[1:].tolist()
N_CLASSES = len(CLASSES)
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
print(f"Classes: {N_CLASSES}, train rows: {len(train_csv)}")

COL_FILE   = "filename"
COL_PRIM   = "primary_label"
COL_SECOND = "secondary_labels"

def balance_index(df):
    rows = []
    for cls, sub in df.groupby(COL_PRIM):
        if len(sub) > CFG["MAX_SAMPLES_PER_CLASS"]:
            sub = sub.sample(CFG["MAX_SAMPLES_PER_CLASS"], random_state=CFG["SEED"])
        if len(sub) < CFG["MIN_SAMPLES_PER_CLASS"]:
            reps = math.ceil(CFG["MIN_SAMPLES_PER_CLASS"] / max(1, len(sub)))
            sub = pd.concat([sub] * reps, ignore_index=True).head(CFG["MIN_SAMPLES_PER_CLASS"])
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)

def detect_group_col(df):
    for c in ("author", "recording_id"):
        if c in df.columns and df[c].astype(str).str.len().gt(0).any():
            return c
    return COL_FILE

train_csv = train_csv[train_csv[COL_PRIM].isin(set(CLASSES))].reset_index(drop=True)
print("Filtered train size:", len(train_csv))

_MEL_TRANSFORM = torchaudio.transforms.MelSpectrogram(
    sample_rate=CFG["SR"], n_fft=CFG["N_FFT"], hop_length=CFG["HOP"],
    n_mels=CFG["N_MELS"], f_min=CFG["FMIN"], f_max=CFG["FMAX"], power=2.0,
)
_DB_TRANSFORM = torchaudio.transforms.AmplitudeToDB(top_db=80)

def wav_to_mel_cpu(wav_np):
    x = torch.from_numpy(wav_np)
    with torch.no_grad():
        m = _MEL_TRANSFORM(x)
        m = _DB_TRANSFORM(m)
        m = (m - m.mean()) / (m.std() + 1e-6)
    return m.unsqueeze(0)

class PinkNoise:
    def __init__(self, prob=0.3, snr_db=(5, 20)):
        self.prob, self.snr_db = prob, snr_db
    def __call__(self, x):
        if random.random() > self.prob: return x
        snr = random.uniform(*self.snr_db)
        n = np.cumsum(np.random.randn(len(x)).astype(np.float32))
        n /= (np.abs(n).max() + 1e-6)
        sig_power = (x ** 2).mean() + 1e-9
        noise_power = sig_power / (10 ** (snr / 10))
        return x + n * math.sqrt(noise_power)

class ExternalNoise:
    def __init__(self, noise_dir, prob=0.4, snr_db=(0, 15)):
        self.files = []
        if noise_dir is not None:
            p = Path(noise_dir)
            if p.exists():
                self.files = sorted([*p.rglob("*.ogg"), *p.rglob("*.wav"),
                                     *p.rglob("*.flac"), *p.rglob("*.mp3")])
        self.prob = prob if self.files else 0.0
        self.snr_db = snr_db
    def __call__(self, x):
        if random.random() > self.prob or not self.files: return x
        sc_path = random.choice(self.files)
        try:
            with sf.SoundFile(str(sc_path)) as f:
                start = random.randint(0, max(0, f.frames - len(x)))
                f.seek(start)
                noise = f.read(len(x), dtype="float32", always_2d=False)
                if noise.ndim == 2: noise = noise.mean(axis=1)
            if len(noise) < len(x):
                noise = np.pad(noise, (0, len(x) - len(noise)))
            noise = noise[:len(x)]
            snr = random.uniform(*self.snr_db)
            sp = (x**2).mean() + 1e-9
            np_pwr = sp / (10 ** (snr/10))
            return x + noise * math.sqrt(np_pwr / ((noise**2).mean() + 1e-9))
        except Exception:
            return x

def spec_augment(spec, time_mask=40, freq_mask=20, n_time=2, n_freq=2, prob=0.8):
    if random.random() > prob: return spec
    if spec.dim() == 3: spec = spec.unsqueeze(1)
    B, _, F_, T = spec.shape
    for _ in range(n_time):
        t = random.randint(0, time_mask)
        if t == 0 or t >= T: continue
        s = random.randint(0, T - t)
        spec[:, :, :, s:s+t] = 0.0
    for _ in range(n_freq):
        f = random.randint(0, freq_mask)
        if f == 0 or f >= F_: continue
        s = random.randint(0, F_ - f)
        spec[:, :, s:s+f, :] = 0.0
    return spec

CLIP_SAMPLES = int(CFG["SR"] * CFG["CLIP_SEC"])

def _shift_zero_pad(x, shift):
    if shift == 0: return x
    out = np.zeros_like(x)
    if shift > 0:
        out[shift:] = x[:len(x) - shift]
    else:
        out[:len(x) + shift] = x[-shift:]
    return out

class FocalAudioDataset(Dataset):
    def __init__(self, df, train_dir, noise_dir, is_train=True):
        self.df = df.reset_index(drop=True)
        self.train_dir = Path(train_dir)
        self.is_train = is_train
        self.pink = PinkNoise(prob=0.3 if is_train else 0.0)
        self.bg_noise = ExternalNoise(noise_dir, prob=0.4 if is_train else 0.0)

    def __len__(self): return len(self.df)

    def _load_5s(self, path):
        try:
            with sf.SoundFile(str(path)) as f:
                total = f.frames
                if total >= CLIP_SAMPLES:
                    start = random.randint(0, total - CLIP_SAMPLES) if self.is_train else 0
                    f.seek(start)
                    wav = f.read(CLIP_SAMPLES, dtype="float32", always_2d=False)
                else:
                    wav = f.read(dtype="float32", always_2d=False)
            if wav.ndim == 2: wav = wav.mean(axis=1)
            wav = wav.astype(np.float32, copy=False)
            if len(wav) > CLIP_SAMPLES:   wav = wav[:CLIP_SAMPLES]
            elif len(wav) < CLIP_SAMPLES: wav = np.pad(wav, (0, CLIP_SAMPLES - len(wav)))
            return wav
        except Exception:
            return np.zeros(CLIP_SAMPLES, dtype=np.float32)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        wav = self._load_5s(self.train_dir / row[COL_FILE])
        if self.is_train:
            shift = random.randint(-CFG["SR"], CFG["SR"])
            wav = _shift_zero_pad(wav, shift)
            wav = wav * random.uniform(0.7, 1.3)
            wav = self.pink(wav)
            wav = self.bg_noise(wav)
        mel = wav_to_mel_cpu(wav)
        y = np.zeros(N_CLASSES, dtype=np.float32)
        y[CLS2IDX[row[COL_PRIM]]] = 1.0
        if COL_SECOND in row and isinstance(row[COL_SECOND], str) and row[COL_SECOND]:
            for s in re.split(r"[,;\s]+", row[COL_SECOND].strip("[]").replace("'", "")):
                if s in CLS2IDX: y[CLS2IDX[s]] = 0.5
        return mel, torch.from_numpy(y)


class PseudoSoundscapeDataset(Dataset):
    def __init__(self, pseudo_df, soundscape_dir, is_train=True):
        self.df = pseudo_df.reset_index(drop=True)
        self.sd = Path(soundscape_dir)
        self.is_train = is_train

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        start = int(row["start_sec"] * CFG["SR"])
        try:
            wav, _ = sf.read(str(self.sd / row["filename"]),
                             start=start, frames=CLIP_SAMPLES,
                             dtype="float32", always_2d=False)
            if wav.ndim == 2: wav = wav.mean(axis=1)
            wav = wav.astype(np.float32, copy=False)
            if len(wav) > CLIP_SAMPLES:   wav = wav[:CLIP_SAMPLES]
            elif len(wav) < CLIP_SAMPLES: wav = np.pad(wav, (0, CLIP_SAMPLES - len(wav)))
        except Exception:
            wav = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        if self.is_train:
            shift = random.randint(-CFG["SR"]//2, CFG["SR"]//2)
            wav = _shift_zero_pad(wav, shift)
            wav = wav * random.uniform(0.8, 1.2)
        mel = wav_to_mel_cpu(wav)
        y = np.array(row["labels"], dtype=np.float32)
        return mel, torch.from_numpy(y)

class AttentionHead(nn.Module):
    def __init__(self, in_feats, n_classes):
        super().__init__()
        self.att = nn.Conv1d(in_feats, n_classes, 1)
        self.cla = nn.Conv1d(in_feats, n_classes, 1)
    def forward(self, x):
        att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla_logits = self.cla(x)
        clip_logits = (att * cla_logits).sum(dim=-1) / (att.sum(dim=-1) + 1e-6)
        return clip_logits, cla_logits

class BirdCNN(nn.Module):
    def __init__(self, backbone, n_classes, use_sed=False):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True,
                                          in_chans=1, num_classes=0, global_pool="")
        feat_dim = self.backbone.num_features
        self.use_sed = use_sed
        if use_sed:
            self.head = AttentionHead(feat_dim, n_classes)
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(feat_dim, n_classes)

    def forward(self, mel):
        f = self.backbone(mel)
        if self.use_sed:
            x = f.mean(dim=2)
            logits, _ = self.head(x)
            return logits, None
        x = self.pool(f).flatten(1)
        return self.fc(x), None

def macro_auc(y_true, y_pred):
    y_bin = (y_true >= 0.99).astype(np.int32)
    keep = y_bin.sum(axis=0) > 0
    if keep.sum() == 0: return 0.0
    return roc_auc_score(y_bin[:, keep], y_pred[:, keep], average="macro")

def mixup_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]

def train_one_fold(fold, train_df, val_df, pseudo_df=None):
    print(f"\n{'='*60}\nFold {fold}  train={len(train_df)}  val={len(val_df)}")
    print(f"  Pseudo data: {len(pseudo_df) if pseudo_df is not None else 0} rows")

    noise_dir = CFG.get("EXTERNAL_NOISE_DIR")
    if noise_dir is not None:
        print(f"  External background-noise dir: {noise_dir}")
    else:
        print(f"  No external noise dir set; ExternalNoise aug disabled.")
    pseudo_sc_dir = BASE / "train_soundscapes"
    tr_ds = FocalAudioDataset(train_df, BASE / "train_audio", noise_dir, is_train=True)
    va_ds = FocalAudioDataset(val_df,   BASE / "train_audio", noise_dir, is_train=False)
    tr_dl = DataLoader(tr_ds, CFG["BATCH_SIZE"], shuffle=True,
                       num_workers=CFG["NUM_WORKERS"], pin_memory=True,
                       drop_last=True, persistent_workers=CFG["NUM_WORKERS"] > 0)
    va_dl = DataLoader(va_ds, CFG["BATCH_SIZE"], shuffle=False,
                       num_workers=CFG["NUM_WORKERS"], pin_memory=True,
                       persistent_workers=CFG["NUM_WORKERS"] > 0)

    pseudo_dl = None
    if pseudo_df is not None and len(pseudo_df) > 0:
        ps_bs = max(1, int(CFG["BATCH_SIZE"] * CFG["PSEUDO_BATCH_RATIO"]))
        ps_ds = PseudoSoundscapeDataset(pseudo_df, pseudo_sc_dir, is_train=True)
        pseudo_dl = DataLoader(ps_ds, ps_bs, shuffle=True,
                               num_workers=CFG["NUM_WORKERS"], pin_memory=True, drop_last=True)
        print(f"  Pseudo batch size: {ps_bs} (focal batch: {CFG['BATCH_SIZE']}, ratio 2:1)")

    model = BirdCNN(CFG["BACKBONE"], N_CLASSES, use_sed=CFG["USE_SED"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["LR"], weight_decay=CFG["WD"])
    steps_per_epoch = len(tr_dl) + (len(pseudo_dl) if pseudo_dl else 0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=CFG["LR"], total_steps=steps_per_epoch * CFG["EPOCHS"],
        pct_start=0.1, anneal_strategy="cos")
    scaler = torch.cuda.amp.GradScaler()
    bce = nn.BCEWithLogitsLoss()

    best_auc, best_state = 0.0, None
    for epoch in range(CFG["EPOCHS"]):
        model.train()
        tloss = 0.0; n = 0
        t0 = time.time()
        iters = zip(tr_dl, pseudo_dl) if pseudo_dl else ((b, None) for b in tr_dl)
        first_batch_logged = False
        for step, batch in enumerate(iters):
            if not first_batch_logged:
                print(f"  epoch {epoch:02d} first batch ({time.time()-t0:.1f}s)")
                first_batch_logged = True
            labelled, pseudo = batch if isinstance(batch, tuple) else (batch, None)
            mel, y = labelled
            mel, y = mel.to(DEVICE), y.to(DEVICE)
            if pseudo is not None:
                pmel, py = pseudo
                mel = torch.cat([mel, pmel.to(DEVICE)], 0)
                y   = torch.cat([y,   py.to(DEVICE)], 0)

            mel = spec_augment(mel)

            if CFG["MIXUP_ALPHA"] > 0 and random.random() < 0.5:
                mel, y = mixup_data(mel, y, CFG["MIXUP_ALPHA"])

            with torch.cuda.amp.autocast():
                logits, _ = model(mel)
                if CFG["LABEL_SMOOTH"] > 0:
                    y = y * (1 - CFG["LABEL_SMOOTH"]) + CFG["LABEL_SMOOTH"] * 0.5
                loss = bce(logits, y)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            sched.step()
            tloss += loss.item() * mel.size(0); n += mel.size(0)

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for mel, y in va_dl:
                mel = mel.to(DEVICE)
                with torch.cuda.amp.autocast():
                    logits, _ = model(mel)
                p = torch.sigmoid(logits.float())
                preds.append(p.cpu().numpy())
                trues.append(y.numpy())
        preds = np.concatenate(preds); trues = np.concatenate(trues)
        auc = macro_auc(trues, preds)
        print(f"  epoch {epoch:02d}  loss={tloss/n:.4f}  val_auc={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    out_pt = Path(CFG["OUT_DIR"]) / f"cnn_{CFG['BACKBONE']}_fold{fold}.pt"
    torch.save(best_state, out_pt)
    print(f"  saved {out_pt}  best_auc={best_auc:.4f}")

    class SigmoidWrap(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, mel):
            logits, _ = self.m(mel)
            return torch.sigmoid(logits)
    export_model = BirdCNN(CFG["BACKBONE"], N_CLASSES, use_sed=CFG["USE_SED"])
    export_model.load_state_dict(best_state)
    export_model.eval().to(DEVICE)
    wrapped = SigmoidWrap(export_model).eval().to(DEVICE)
    dummy = torch.randn(1, 1, CFG["N_MELS"], CFG["IMG_SIZE"][1], device=DEVICE)
    out_onnx = Path(CFG["OUT_DIR"]) / f"cnn_{CFG['BACKBONE']}_fold{fold}.onnx"
    torch.onnx.export(wrapped, dummy, str(out_onnx),
                      input_names=["mel"], output_names=["prob"],
                      dynamic_axes={"mel": {0: "B"}, "prob": {0: "B"}},
                      opset_version=17, dynamo=False)
    print(f"  exported {out_onnx}")

    del model, tr_dl, va_dl, pseudo_dl; gc.collect(); torch.cuda.empty_cache()
    return best_auc

GROUP_COL = detect_group_col(train_csv)
print(f"CV grouping key: {GROUP_COL}")
groups = train_csv[GROUP_COL].astype(str).fillna("")
groups = groups.where(groups != "", train_csv[COL_FILE].astype(str)).str.lower().values
sgkf = StratifiedGroupKFold(n_splits=CFG["N_FOLDS"], shuffle=True, random_state=CFG["SEED"])
y_strat = train_csv[COL_PRIM].values
folds = list(sgkf.split(train_csv, y_strat, groups=groups))
for f, (tr, va) in enumerate(folds):
    overlap = set(groups[tr].tolist()) & set(groups[va].tolist())
    assert not overlap, f"Fold {f}: {len(overlap)} groups leak across train/val"
print(f"Created {len(folds)} folds with no group overlap.")

pseudo_df = None
if CFG["PSEUDO_DIR"] is not None:
    pseudo_path = next((Path(CFG["PSEUDO_DIR"]).glob("pseudo_labels.parquet")), None)
    if pseudo_path is None:
        cands = list(Path("/kaggle/input").rglob("pseudo_labels.parquet"))
        if cands:
            pseudo_path = cands[0]
    if pseudo_path and pseudo_path.exists():
        pseudo_df = pd.read_parquet(pseudo_path)
        print(f"\nLoaded pseudo-labels: {len(pseudo_df)} rows from {pseudo_path}")
    else:
        print(f"\nPseudo-labels not found. Falling back to focal-only training.")

aucs = []
for f, (tr_idx, va_idx) in enumerate(folds):
    if f not in CFG["FOLDS_TO_RUN"]: continue
    tr_df = balance_index(train_csv.iloc[tr_idx])
    va_df = train_csv.iloc[va_idx].reset_index(drop=True)
    print(f"Fold {f}: train rows {len(tr_df)} (balanced from {len(tr_idx)}), val rows {len(va_df)}")
    auc = train_one_fold(f, tr_df, va_df, pseudo_df)
    aucs.append(auc)
print(f"\nAll folds val AUC: {aucs} mean: {np.mean(aucs) if aucs else 0.0}")
