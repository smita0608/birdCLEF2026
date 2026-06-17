from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/user/work/tf-mp/kaggle/birdclef_upgrade')
from soundscape_val import build_soundscape_val_meta, score_predictions

WORK = Path('/home/user/work/tf-mp/kaggle/working_949')
BASE = Path('/home/user/work/tf-mp/kaggle/birdclef-2026')


def main():
    classes = pd.read_csv(BASE / 'sample_submission.csv', nrows=1).columns[1:].tolist()
    meta = build_soundscape_val_meta(BASE, classes=classes,
                                      only_fully_labeled=True, n_windows_per_file=12)
    y_true = np.stack(meta['labels'].to_numpy()).astype(np.float32)

    proto = np.load(WORK / 'oof_proto_5fold.npy')
    sed = np.load(WORK / 'oof_sed_5fold.npy')
    print(f'Loaded OOF preds: proto {proto.shape}, sed {sed.shape}')

    n = len(meta)
    n_classes = len(classes)

    proto_rank = np.zeros_like(proto, dtype=np.float32)
    sed_rank = np.zeros_like(sed, dtype=np.float32)
    for c in range(n_classes):
        proto_rank[:, c] = pd.Series(proto[:, c]).rank(method='average').to_numpy() / n
        sed_rank[:, c] = pd.Series(sed[:, c]).rank(method='average').to_numpy() / n

    grid = np.linspace(0.0, 1.0, 21)
    weights = np.full(n_classes, 0.6, dtype=np.float32)
    active = (y_true.sum(axis=0) >= 5)
    n_active = int(active.sum())
    print(f'\nFitting per-class blend weight on {n_active} active classes (≥5 positives)')

    for c in range(n_classes):
        if not active[c]:
            continue
        yc = y_true[:, c]
        best_auc, best_w = -1.0, 0.6
        for w in grid:
            blend = w * proto_rank[:, c] + (1 - w) * sed_rank[:, c]
            try:
                a = roc_auc_score(yc, blend)
            except Exception:
                continue
            if a > best_auc:
                best_auc, best_w = a, float(w)
        weights[c] = best_w

    blended = weights[None, :] * proto_rank + (1 - weights)[None, :] * sed_rank
    auc_perclass = score_predictions(blended, meta)

    blended_default = 0.6 * proto_rank + 0.4 * sed_rank
    auc_default = score_predictions(blended_default, meta)

    e1_path = WORK / 'e1_blend_sweep_result.json'
    if e1_path.exists():
        e1 = json.loads(e1_path.read_text())
        wg = e1['best_blend_weight_proto']
        blended_e1 = wg * proto_rank + (1 - wg) * sed_rank
        auc_e1_global = score_predictions(blended_e1, meta)
    else:
        auc_e1_global = None
        wg = None

    print(f'\nResults:')
    print(f'  Default 60/40 global: AUC = {auc_default:.5f}')
    if auc_e1_global is not None:
        print(f'  E1 best global ({wg}/{1-wg}): AUC = {auc_e1_global:.5f}')
    print(f'  Per-class weights:    AUC = {auc_perclass:.5f}')

    print(f'\nPer-class weight distribution (active classes only):')
    print(f'  proto_w mean = {weights[active].mean():.3f}')
    print(f'  proto_w std  = {weights[active].std():.3f}')
    print(f'  proto_w pctiles: 5% = {np.percentile(weights[active], 5):.2f}, '
          f'25% = {np.percentile(weights[active], 25):.2f}, '
          f'50% = {np.percentile(weights[active], 50):.2f}, '
          f'75% = {np.percentile(weights[active], 75):.2f}, '
          f'95% = {np.percentile(weights[active], 95):.2f}')

    np.save(WORK / 'per_class_blend_weights.npy', weights)
    delta = float(auc_perclass - (auc_e1_global if auc_e1_global is not None else auc_default))
    print(f'\n  delta vs {"E1 best global" if auc_e1_global is not None else "60/40"}: {delta:+.5f}')

    result = {
        'experiment': 'E4_per_class_blend',
        'auc_default_60_40': float(auc_default),
        'auc_e1_best_global': float(auc_e1_global) if auc_e1_global is not None else None,
        'auc_per_class': float(auc_perclass),
        'n_active_classes': n_active,
        'proto_weight_mean': float(weights[active].mean()),
        'proto_weight_std': float(weights[active].std()),
        'delta_vs_best_global': delta,
    }
    (WORK / 'e4_per_class_result.json').write_text(json.dumps(result, indent=2))

    if delta >= 0.003:
        print(f'  ✓ STRONG candidate (>+0.003)')
    elif delta >= 0.001:
        print(f'  ~ MARGINAL (need retest)')
    else:
        print(f'  ✗ NEGLIGIBLE')


if __name__ == '__main__':
    main()
