from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path('/home/user/work/tf-mp/kaggle/birdclef-2026')
WORK = Path('/home/user/work/tf-mp/kaggle/working_949')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=0.3,
                    help='Keep rows where row.max() > threshold (Sydorskyy uses 0.5)')
    ap.add_argument('--trim', type=float, default=0.1,
                    help='Set values < trim to 0 (Sydorskyy spec)')
    ap.add_argument('--proto_weight', type=float, default=0.6)
    ap.add_argument('--sed_weight', type=float, default=0.4)
    ap.add_argument('--no_birdnet', action='store_true',
                    help='Ablation: combine only Perch+SED, no BirdNET')
    ap.add_argument('--out_parquet', default=str(WORK / 'pseudo_labels_v2.parquet'))
    ap.add_argument('--pseudo_dir', default=str(WORK / 'pseudo_gen'),
                    help='Directory containing the 3 teacher CSVs (default: pseudo_gen)')
    args = ap.parse_args()

    classes = pd.read_csv(BASE / 'sample_submission.csv', nrows=1).columns[1:].tolist()
    N_CLASSES = len(classes)
    assert N_CLASSES == 234

    pg = Path(args.pseudo_dir)
    proto_csv = pg / 'submission_protossm.csv'
    sed_csv = pg / 'submission_sed.csv'
    bn_csv = pg / 'submission_birdnet.csv'

    assert proto_csv.exists(), f'Missing {proto_csv}'
    assert sed_csv.exists(),   f'Missing {sed_csv}'

    use_birdnet = (not args.no_birdnet) and bn_csv.exists()
    if not use_birdnet:
        print('NOTE: BirdNET CSV not found or disabled — using Perch+SED only')

    print(f'Loading Perch (proto)... ', end='', flush=True)
    df_proto = pd.read_csv(proto_csv)
    print(f'{df_proto.shape}')
    print(f'Loading Tucker SED...    ', end='', flush=True)
    df_sed = pd.read_csv(sed_csv)
    print(f'{df_sed.shape}')

    df_sed = df_sed.set_index('row_id').loc[df_proto['row_id']].reset_index()
    assert (df_proto['row_id'].values == df_sed['row_id'].values).all()

    p_proto = df_proto[classes].to_numpy(dtype=np.float32)
    p_sed   = df_sed[classes].to_numpy(dtype=np.float32)
    teacher_ps = args.proto_weight * p_proto + args.sed_weight * p_sed
    print(f'Perch+SED blend: range [{teacher_ps.min():.4f}, {teacher_ps.max():.4f}]')

    if use_birdnet:
        print(f'Loading BirdNET...       ', end='', flush=True)
        df_bn = pd.read_csv(bn_csv)
        print(f'{df_bn.shape}')
        df_bn = df_bn.set_index('row_id').loc[df_proto['row_id']].reset_index()
        assert (df_proto['row_id'].values == df_bn['row_id'].values).all()
        p_bn = df_bn[classes].to_numpy(dtype=np.float32)
        print(f'BirdNET: range [{p_bn.min():.4f}, {p_bn.max():.4f}]')

        combined = np.maximum(teacher_ps, p_bn)
        print(f'Combined max(PS, BN): range [{combined.min():.4f}, {combined.max():.4f}], '
              f'mean {combined.mean():.4f}')
    else:
        combined = teacher_ps

    row_max = combined.max(axis=1)
    keep_mask = row_max > args.threshold
    print(f'\nFilter rows where max > {args.threshold}: '
          f'{keep_mask.sum()}/{len(keep_mask)} ({100*keep_mask.mean():.1f}%)')
    if keep_mask.sum() == 0:
        raise RuntimeError(f'No rows survive threshold {args.threshold}')

    combined_kept = combined[keep_mask]
    row_ids_kept = df_proto['row_id'].values[keep_mask]

    combined_kept = combined_kept.copy()
    combined_kept[combined_kept < args.trim] = 0.0
    print(f'Trim values < {args.trim}: nonzero fraction = '
          f'{(combined_kept > 0).mean():.4f}')

    primary_idx = combined_kept.argmax(axis=1)
    primary_prob = combined_kept.max(axis=1)
    primary_label = [classes[i] for i in primary_idx]
    filenames = []
    start_secs = []
    for rid in row_ids_kept:
        parts = rid.rsplit('_', 1)
        stem = parts[0]
        end_sec = int(parts[1])
        filenames.append(stem + '.ogg')
        start_secs.append(end_sec - 5)

    out = pd.DataFrame({
        'filename':           filenames,
        'start_sec':          np.array(start_secs, dtype=np.int32),
        'primary_label':      primary_label,
        'primary_label_prob': primary_prob.astype(np.float32),
        'labels':             list(combined_kept.astype(np.float32)),
    })
    print(f'\nOutput shape: {out.shape}')

    classes_with_any = (combined_kept > 0).any(axis=0).sum()
    classes_with_5plus = (combined_kept > 0).sum(axis=0) >= 5
    classes_with_5plus_count = classes_with_5plus.sum()
    print(f'Classes with any nonzero label: {classes_with_any}/{N_CLASSES}')
    print(f'Classes with ≥5 nonzero labels: {classes_with_5plus_count}/{N_CLASSES}')

    primary_counts = pd.Series(primary_label).value_counts()
    print(f'Classes as primary_label (top 5): {primary_counts.head(5).to_dict()}')
    print(f'Total classes appearing as primary: {len(primary_counts)}/{N_CLASSES}')

    out.to_parquet(args.out_parquet, index=False)
    print(f'\nWrote {args.out_parquet} ({Path(args.out_parquet).stat().st_size / 1e6:.1f} MB)')

    meta = {
        'threshold':                       args.threshold,
        'trim':                            args.trim,
        'proto_weight':                    args.proto_weight,
        'sed_weight':                      args.sed_weight,
        'used_birdnet':                    bool(use_birdnet),
        'n_input_rows':                    int(len(df_proto)),
        'n_kept_rows':                     int(keep_mask.sum()),
        'keep_ratio':                      float(keep_mask.mean()),
        'n_classes_with_any_nonzero':      int(classes_with_any),
        'n_classes_with_5plus_nonzero':    int(classes_with_5plus_count),
        'n_classes_as_primary':            int(len(primary_counts)),
    }
    (WORK / 'e13_combine_pseudos_result.json').write_text(json.dumps(meta, indent=2))
    print(f'Meta saved to e13_combine_pseudos_result.json')


if __name__ == '__main__':
    main()
