from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/user/work/tf-mp/kaggle/birdclef_upgrade')
from soundscape_val import build_soundscape_val_meta

WORK = Path('/home/user/work/tf-mp/kaggle/working_949')
BASE = Path('/home/user/work/tf-mp/kaggle/birdclef-2026')
PIPELINE = WORK / 'pipeline.py'

N_SAMPLE_FILES = 2000
MID_WINDOWS = [4, 5, 6, 7]
TOPK = 3
CONF_THRESHOLD = 0.5
SOFT_FLOOR = 0.01
SEED = 42
PROTO_W = 0.6
SED_W = 0.4
N_WINDOWS = 12


def select_unlabeled_files() -> list[str]:
    classes = pd.read_csv(BASE / 'sample_submission.csv', nrows=1).columns[1:].tolist()
    meta = build_soundscape_val_meta(BASE, classes=classes,
                                      only_fully_labeled=False, n_windows_per_file=12)
    labeled_files = set(meta['filename'].unique())
    print(f'  {len(labeled_files)} files in labels CSV')

    all_files = sorted([p.name for p in (BASE / 'train_soundscapes').glob('*.ogg')])
    print(f'  {len(all_files)} total soundscape files on disk')

    unlabeled = [f for f in all_files if f not in labeled_files]
    print(f'  {len(unlabeled)} unlabeled candidates')

    rng = random.Random(SEED)
    rng.shuffle(unlabeled)
    sampled = sorted(unlabeled[:N_SAMPLE_FILES])
    print(f'  sampled {len(sampled)} files')
    return sampled


def run_pipeline_on_files(file_list: list[str], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(exist_ok=True, parents=True)
    file_list_path = out_dir / 'pseudo_test_files.txt'
    file_list_path.write_text('\n'.join(file_list) + '\n')
    print(f'  Wrote file list: {file_list_path} ({len(file_list)} files)')

    env = os.environ.copy()
    env['FOLD_TEST_FILES_PATH'] = str(file_list_path)
    env['FOLD_WORK_DIR'] = str(out_dir / 'cache')
    env['PYTHONUNBUFFERED'] = '1'

    venv_python = WORK.parent / '.venv' / 'bin' / 'python'
    log_path = out_dir / 'pseudo_gen.log'
    print(f'  Running pipeline.py — log: {log_path}')
    t0 = time.time()
    with open(log_path, 'w') as logf:
        proc = subprocess.run(
            [str(venv_python), str(PIPELINE)],
            cwd=str(out_dir),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    elapsed = (time.time() - t0) / 60
    print(f'  pipeline finished in {elapsed:.1f} min  rc={proc.returncode}')
    if proc.returncode != 0:
        with open(log_path) as f:
            tail = f.readlines()[-30:]
        print('  TAIL:')
        for line in tail:
            print('   ', line.rstrip())
        raise RuntimeError(f'pseudo-gen pipeline failed rc={proc.returncode}')

    proto = out_dir / 'submission_protossm.csv'
    sed = out_dir / 'submission_sed.csv'
    if not proto.exists() or not sed.exists():
        raise FileNotFoundError(f'expected CSVs missing: {proto} / {sed}')
    return proto, sed


def rank_blend(a: np.ndarray, b: np.ndarray, wa: float = PROTO_W, wb: float = SED_W) -> np.ndarray:
    out = np.zeros_like(a, dtype=np.float32)
    n = len(a)
    for c in range(a.shape[1]):
        ra = pd.Series(a[:, c]).rank(method='average').to_numpy() / n
        rb = pd.Series(b[:, c]).rank(method='average').to_numpy() / n
        out[:, c] = wa * ra + wb * rb
    return out


def extract_pseudo_rows(proto_csv: Path, sed_csv: Path) -> pd.DataFrame:
    classes = pd.read_csv(BASE / 'sample_submission.csv', nrows=1).columns[1:].tolist()
    n_classes = len(classes)

    proto_df = pd.read_csv(proto_csv)
    sed_df = pd.read_csv(sed_csv)
    print(f'  proto: {proto_df.shape}, sed: {sed_df.shape}')

    proto_df = proto_df.set_index('row_id')
    sed_df = sed_df.set_index('row_id')
    common = sorted(set(proto_df.index) & set(sed_df.index))
    proto = proto_df.loc[common, classes].to_numpy(dtype=np.float32)
    sed = sed_df.loc[common, classes].to_numpy(dtype=np.float32)

    print(f'  prob-blending {len(common)} rows @ {PROTO_W}/{SED_W}...')
    teacher = (PROTO_W * proto + SED_W * sed).astype(np.float32)

    row_ids = pd.Series(common)
    parts = row_ids.str.rsplit('_', n=1, expand=True)
    parts.columns = ['stem', 'end_sec']
    parts['end_sec'] = parts['end_sec'].astype(int)
    parts['window_id'] = parts['end_sec'] // 5 - 1
    parts['filename'] = parts['stem'] + '.ogg'

    keep_mask = parts['window_id'].isin(MID_WINDOWS).to_numpy()
    print(f'  mid-window filter: {keep_mask.sum()}/{len(keep_mask)} rows')
    teacher = teacher[keep_mask]
    parts = parts.loc[keep_mask].reset_index(drop=True)

    rows = []
    n_skipped = 0
    for i in range(len(parts)):
        probs = teacher[i]
        order = np.argsort(-probs)[:TOPK]
        y = np.full(n_classes, SOFT_FLOOR, dtype=np.float32)
        n_kept = 0
        for k in order:
            if probs[k] >= CONF_THRESHOLD:
                y[k] = float(probs[k])
                n_kept += 1
        if n_kept == 0:
            n_skipped += 1
            continue
        rows.append({
            'filename': parts.iloc[i]['filename'],
            'start_sec': int(parts.iloc[i]['window_id']) * 5,
            'labels': y.tolist(),
        })
    print(f'  TOPK+conf filter: kept {len(rows)} rows, skipped {n_skipped} all-low-conf rows')

    return pd.DataFrame(rows)


def main():
    print(f'=== E2.1 — Mid-window pseudo-label generation ===')
    print(f'Sampling {N_SAMPLE_FILES} unlabeled soundscapes...')
    files = select_unlabeled_files()

    pseudo_dir = WORK / 'pseudo_gen'
    proto_csv, sed_csv = run_pipeline_on_files(files, pseudo_dir)

    print(f'\nExtracting pseudo-labels...')
    pdf = extract_pseudo_rows(proto_csv, sed_csv)

    out = WORK / 'pseudo_labels.parquet'
    pdf.to_parquet(out, index=False)
    print(f'\nSaved {out}: {len(pdf)} rows')

    n_files_kept = pdf['filename'].nunique()
    print(f'  unique files: {n_files_kept}')
    print(f'  mean kept windows per file: {len(pdf) / max(1, n_files_kept):.1f}')
    keep_ratio = len(pdf) / max(1, N_SAMPLE_FILES * len(MID_WINDOWS))
    print(f'  keep ratio: {keep_ratio*100:.1f}% (target 30-95%)')

    result = {
        'experiment': 'E2.1_pseudo_gen',
        'n_files_sampled': N_SAMPLE_FILES,
        'n_unique_files_kept': int(n_files_kept),
        'n_pseudo_rows': int(len(pdf)),
        'keep_ratio': float(keep_ratio),
        'mid_windows': MID_WINDOWS,
        'top_k': TOPK,
        'conf_threshold': CONF_THRESHOLD,
        'teacher_blend': {'proto': PROTO_W, 'sed': SED_W},
    }
    (WORK / 'e2_pseudo_gen_result.json').write_text(json.dumps(result, indent=2))
    print(f'  metadata saved to {WORK / "e2_pseudo_gen_result.json"}')


if __name__ == '__main__':
    main()
