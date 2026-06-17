from __future__ import annotations

import json
import os
import runpy
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, '/home/user/work/tf-mp/kaggle/birdclef_upgrade')
from soundscape_val import build_soundscape_val_meta, score_predictions

WORK = Path('/home/user/work/tf-mp/kaggle/working_949')
BASE = Path('/home/user/work/tf-mp/kaggle/birdclef-2026')
PIPELINE = WORK / 'pipeline.py'
N_FOLDS = 5
SEED = 42


def build_fold_splits() -> tuple[pd.DataFrame, List[List[str]], List[List[str]]]:
    classes = pd.read_csv(BASE / 'sample_submission.csv', nrows=1).columns[1:].tolist()
    meta = build_soundscape_val_meta(BASE, classes=classes,
                                      only_fully_labeled=True, n_windows_per_file=12)
    files = sorted(meta['filename'].unique().tolist())
    print(f'{len(files)} fully-labeled soundscape files; splitting into {N_FOLDS} folds')
    splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    trains, tests = [], []
    for fold_k, (tr_idx, te_idx) in enumerate(splitter.split(files)):
        tr = sorted([files[i] for i in tr_idx])
        te = sorted([files[i] for i in te_idx])
        trains.append(tr)
        tests.append(te)
        print(f'  fold {fold_k}: train={len(tr)}, test={len(te)} ({te[:2]}{"..." if len(te)>2 else ""})')
    return meta, trains, tests


def run_fold(fold_k: int, train_files: List[str], test_files: List[str]) -> Path:
    fold_work = WORK / f'fold_{fold_k}_cache'
    fold_work.mkdir(exist_ok=True, parents=True)

    fold_run = WORK / f'fold_{fold_k}_run'
    fold_run.mkdir(exist_ok=True, parents=True)

    env = os.environ.copy()
    env['FOLD_TRAIN_FILES'] = ','.join(train_files)
    env['FOLD_TEST_FILES'] = ','.join(test_files)
    env['FOLD_WORK_DIR'] = str(fold_work)
    env['PYTHONUNBUFFERED'] = '1'

    log_path = fold_run / 'pipeline.log'
    print(f'\n{"="*60}\nFold {fold_k}/{N_FOLDS} — training on {len(train_files)} files, '
          f'predicting on {len(test_files)}\n  WORK_DIR={fold_work}\n  LOG={log_path}\n{"="*60}')

    t0 = time.time()
    venv_python = WORK.parent / '.venv' / 'bin' / 'python'
    with open(log_path, 'w') as logf:
        proc = subprocess.run(
            [str(venv_python), str(PIPELINE)],
            cwd=str(fold_run),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    elapsed = (time.time() - t0) / 60
    print(f'  fold {fold_k} finished in {elapsed:.1f} min  rc={proc.returncode}')

    if proc.returncode != 0:
        with open(log_path) as f:
            tail = f.readlines()[-30:]
        print('  TAIL OF LOG:')
        for line in tail:
            print('   ', line.rstrip())
        raise RuntimeError(f'fold {fold_k} pipeline.py failed (rc={proc.returncode})')

    candidates = [
        fold_run / 'submission_protossm.csv',
        fold_run / 'submission.csv',
        fold_run / 'subm_karnakbayev_power_optimization.csv',
    ]
    for cand in candidates:
        if cand.exists():
            print(f'  fold {fold_k} prediction CSV: {cand.name}')
            return cand
    raise FileNotFoundError(f'fold {fold_k} produced no submission CSV in {fold_run}')


def aggregate_folds(meta: pd.DataFrame, fold_test_files: List[List[str]],
                    fold_csvs: List[Path]) -> np.ndarray:
    classes = pd.read_csv(BASE / 'sample_submission.csv', nrows=1).columns[1:].tolist()
    meta = meta.copy()
    meta['row_id'] = (
        meta['filename'].str.replace('.ogg', '', regex=False)
        + '_' + meta['end_sec'].astype(str)
    )
    preds = np.zeros((len(meta), len(classes)), dtype=np.float32)
    seen = np.zeros(len(meta), dtype=bool)
    for k, csv in enumerate(fold_csvs):
        sub = pd.read_csv(csv).set_index('row_id')
        sub = sub[classes]
        fold_files = set(fold_test_files[k])
        fold_mask = meta['filename'].isin(fold_files).to_numpy()
        if seen[fold_mask].any():
            overlap = (seen & fold_mask).sum()
            print(f'  WARNING: fold {k} overlaps {overlap} already-filled rows')
        target_row_ids = meta.loc[fold_mask, 'row_id'].tolist()
        missing = [r for r in target_row_ids if r not in sub.index]
        if missing:
            print(f'  fold {k}: {len(missing)} held-out row_ids missing from sub; '
                  f'first 3: {missing[:3]}')
        aligned = sub.reindex(target_row_ids)
        preds[fold_mask] = aligned[classes].to_numpy(dtype=np.float32)
        seen[fold_mask] = True
    unfilled = (~seen).sum()
    if unfilled:
        print(f'  WARNING: {unfilled} meta rows never filled — left as 0')
    return preds


def main():
    print(f'5-fold OOF run @ {WORK}\n  pipeline: {PIPELINE}')
    if not PIPELINE.exists():
        raise FileNotFoundError(f'{PIPELINE} not found — run extract_pipeline.py first')

    meta, trains, tests = build_fold_splits()

    split_path = WORK / 'kfold_split.json'
    split_path.write_text(json.dumps({'trains': trains, 'tests': tests}, indent=2))
    print(f'Persisted split to {split_path}')

    t_all = time.time()
    fold_csvs = []
    for k, (tr, te) in enumerate(zip(trains, tests)):
        csv = run_fold(k, tr, te)
        fold_csvs.append(csv)
    print(f'\nAll 5 folds done in {(time.time() - t_all)/60:.1f} min')

    preds = aggregate_folds(meta, tests, fold_csvs)
    np.save(WORK / 'oof_5fold_preds.npy', preds)

    auc = score_predictions(preds, meta)
    print(f'\n  ===> local_auc_oof_5fold = {auc:.5f}')

    out = WORK / 'kfold_result.json'
    out.write_text(json.dumps({
        'local_auc_oof_5fold': float(auc),
        'n_windows': len(meta),
        'fold_csvs': [str(p) for p in fold_csvs],
        'split_seed': SEED,
        'n_folds': N_FOLDS,
    }, indent=2))
    print(f'Saved result to {out}')


if __name__ == '__main__':
    main()
