from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path('/home/user/work/tf-mp/kaggle/working_949')
BASE = Path('/home/user/work/tf-mp/kaggle/birdclef-2026')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gamma', type=float, default=2.0, help='Power scaling exponent (Sydorskyy: 2-4)')
    ap.add_argument('--trim', type=float, default=0.1, help='Set values < trim to 0 (Sydorskyy spec)')
    ap.add_argument('--in_parquet', default=str(WORK / 'pseudo_labels_v2.parquet'))
    ap.add_argument('--out_parquet', default=None)
    args = ap.parse_args()

    if args.out_parquet is None:
        gamma_tag = str(args.gamma).replace('.', '').rstrip('0') or '0'
        args.out_parquet = str(WORK / f'pseudo_labels_v2_power{gamma_tag}.parquet')

    classes = pd.read_csv(BASE / 'sample_submission.csv', nrows=1).columns[1:].tolist()
    N_CLASSES = len(classes)

    print(f'Loading {args.in_parquet}')
    df = pd.read_parquet(args.in_parquet)
    print(f'  shape: {df.shape}')
    print(f'  columns: {list(df.columns)}')

    labels_arr = np.stack(df['labels'].to_numpy()).astype(np.float32)
    assert labels_arr.shape == (len(df), N_CLASSES), f'Bad labels shape: {labels_arr.shape}'

    print(f'\nBefore Power Scaling (γ={args.gamma}):')
    print(f'  values range: [{labels_arr.min():.4f}, {labels_arr.max():.4f}]')
    print(f'  nonzero fraction: {(labels_arr > 0).mean():.4f}')
    print(f'  values > 0.5 fraction: {(labels_arr > 0.5).mean():.4f}')

    sharpened = labels_arr ** args.gamma

    sharpened[sharpened < args.trim] = 0.0
    sharpened = sharpened.astype(np.float32)

    print(f'\nAfter Power Scaling + trim:')
    print(f'  values range: [{sharpened.min():.4f}, {sharpened.max():.4f}]')
    print(f'  nonzero fraction: {(sharpened > 0).mean():.4f}')
    print(f'  values > 0.5 fraction: {(sharpened > 0.5).mean():.4f}')

    primary_idx = sharpened.argmax(axis=1)
    primary_prob = sharpened.max(axis=1)
    primary_label = [classes[i] for i in primary_idx]

    n_zero_rows = (primary_prob == 0).sum()
    if n_zero_rows > 0:
        print(f'\n⚠️  {n_zero_rows} rows now have ALL-ZERO labels (lost via Power Scaling + trim)')
        print(f'   Consider lowering γ or trim threshold if many.')

    keep_mask = primary_prob > 0
    sharpened = sharpened[keep_mask]
    primary_idx = primary_idx[keep_mask]
    primary_prob = primary_prob[keep_mask]
    primary_label = [primary_label[i] for i in range(len(primary_label)) if keep_mask[i]]

    out = pd.DataFrame({
        'filename':           df['filename'].values[keep_mask],
        'start_sec':          df['start_sec'].values[keep_mask].astype(np.int32),
        'primary_label':      primary_label,
        'primary_label_prob': primary_prob.astype(np.float32),
        'labels':             list(sharpened),
    })
    print(f'\nOutput shape: {out.shape} (after dropping {n_zero_rows} zero-row records)')

    classes_with_any = (sharpened > 0).any(axis=0).sum()
    classes_with_5plus = ((sharpened > 0).sum(axis=0) >= 5).sum()
    primary_counts = pd.Series(primary_label).value_counts()
    print(f'Classes with any nonzero label:  {classes_with_any}/{N_CLASSES}')
    print(f'Classes with ≥5 nonzero labels:  {classes_with_5plus}/{N_CLASSES}')
    print(f'Classes appearing as primary:    {len(primary_counts)}/{N_CLASSES}')
    print(f'Top 5 primary: {primary_counts.head(5).to_dict()}')

    out.to_parquet(args.out_parquet, index=False)
    print(f'\nWrote {args.out_parquet} ({Path(args.out_parquet).stat().st_size / 1e6:.1f} MB)')

    meta = {
        'gamma':                          args.gamma,
        'trim':                           args.trim,
        'in_parquet':                     args.in_parquet,
        'out_parquet':                    args.out_parquet,
        'n_input_rows':                   int(len(df)),
        'n_output_rows':                  int(len(out)),
        'n_zero_rows_dropped':            int(n_zero_rows),
        'n_classes_with_any_nonzero':     int(classes_with_any),
        'n_classes_with_5plus_nonzero':   int(classes_with_5plus),
        'n_classes_as_primary':           int(len(primary_counts)),
    }
    meta_path = WORK / f'e19_power_scale_result_g{args.gamma}.json'
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f'Metadata: {meta_path}')


if __name__ == '__main__':
    main()
