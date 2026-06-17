from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


_FILENAME_RE = re.compile(r"^([A-Za-z0-9]+)_([A-Z]+)_(\d{8})_(\d{6})\.ogg$")


def parse_soundscape_filename(name: str) -> dict:
    m = _FILENAME_RE.match(name)
    if m:
        file_id, site, ymd, hms = m.groups()
        return {"file_id": file_id, "site": site, "ymd": ymd, "hms": hms}
    return {"file_id": Path(name).stem, "site": "", "ymd": "", "hms": ""}


def build_soundscape_val_meta(
    base_dir: str | Path,
    classes: Optional[Iterable[str]] = None,
    only_fully_labeled: bool = True,
    n_windows_per_file: int = 12,
) -> pd.DataFrame:
    base = Path(base_dir)
    csv = base / "train_soundscapes_labels.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found. Soundscape labels are required for held-out val."
        )

    df = pd.read_csv(csv)
    if "end_sec" not in df.columns:
        if "end" in df.columns:
            df = df.rename(columns={"end": "end_sec"})
        else:
            raise KeyError(
                f"Expected end_sec / end column in {csv}; got {df.columns.tolist()}"
            )
    if "primary_label" not in df.columns:
        raise KeyError("Expected primary_label column in soundscape labels CSV")
    df["primary_label"] = df["primary_label"].astype(str)
    if df["end_sec"].dtype == object:
        df["end_sec"] = pd.to_timedelta(df["end_sec"]).dt.total_seconds().astype(int)
    else:
        df["end_sec"] = df["end_sec"].astype(int)

    if classes is None:
        classes = sorted(df["primary_label"].unique().tolist())
    classes = list(classes)
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    def _collect_labels(s):
        out = set()
        for entry in s:
            if pd.isna(entry):
                continue
            for tok in str(entry).split(";"):
                tok = tok.strip()
                if tok and tok in cls_to_idx:
                    out.add(tok)
        return list(out)

    agg = (
        df.groupby(["filename", "end_sec"])["primary_label"]
        .apply(_collect_labels)
        .reset_index()
        .rename(columns={"primary_label": "labels_list"})
    )

    if only_fully_labeled:
        windows_per_file = agg.groupby("filename").size()
        keep = windows_per_file[windows_per_file >= n_windows_per_file].index
        agg = agg[agg["filename"].isin(keep)].reset_index(drop=True)

    label_mat = np.zeros((len(agg), n_classes), dtype=np.float32)
    for i, labs in enumerate(agg["labels_list"].values):
        for c in labs:
            label_mat[i, cls_to_idx[c]] = 1.0

    agg["labels"] = list(label_mat)
    agg["window_id"] = (agg["end_sec"] // 5 - 1).astype(int)
    agg["start_sec"] = agg["window_id"] * 5
    meta_cols = agg["filename"].apply(parse_soundscape_filename).apply(pd.Series)
    out = pd.concat([agg, meta_cols], axis=1)
    out = out.sort_values(["filename", "window_id"]).reset_index(drop=True)
    return out


def score_predictions(
    predictions: np.ndarray,
    meta: pd.DataFrame,
    n_classes: Optional[int] = None,
) -> float:
    n_classes = n_classes or predictions.shape[1]
    y_true = np.stack(meta["labels"].to_numpy()).astype(np.float32)
    if y_true.shape != predictions.shape:
        raise ValueError(
            f"predictions shape {predictions.shape} != labels shape {y_true.shape}; "
            f"row alignment between meta and predictions is required."
        )
    keep = y_true.sum(axis=0) > 0
    if keep.sum() == 0:
        return 0.0
    return float(
        roc_auc_score(y_true[:, keep], predictions[:, keep], average="macro")
    )


def split_by_site(meta: pd.DataFrame) -> dict:
    out = {}
    for site, sub in meta.groupby("site"):
        out[site or "UNKNOWN"] = sub.reset_index(drop=True)
    return out


__all__ = [
    "parse_soundscape_filename",
    "build_soundscape_val_meta",
    "score_predictions",
    "split_by_site",
]
