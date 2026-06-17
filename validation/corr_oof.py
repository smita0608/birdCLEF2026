from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr

WORK = Path("/home/user/work/tf-mp/kaggle/working_949")


def per_class_corr(a, b, fn):
    vals = []
    for c in range(a.shape[1]):
        ac, bc = a[:, c], b[:, c]
        if np.std(ac) < 1e-9 or np.std(bc) < 1e-9:
            continue
        r = fn(ac, bc).correlation if fn is spearmanr else fn(ac, bc)[0]
        if np.isfinite(r):
            vals.append(r)
    return np.array(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(WORK / "sed_v2s_e14_full_out/oof_preds_best.npy"))
    ap.add_argument("--b", required=True)
    ap.add_argument("--name_a", default="V2-s")
    ap.add_argument("--name_b", default="model-B")
    args = ap.parse_args()

    a = np.load(args.a).astype(np.float64)
    b = np.load(args.b).astype(np.float64)
    print(f"{args.name_a}: {a.shape}   {args.name_b}: {b.shape}")
    assert a.shape == b.shape, "OOF shapes differ — not the same heldout split/order"

    flat_p = pearsonr(a.ravel(), b.ravel())[0]

    pc_pear = per_class_corr(a, b, pearsonr)
    pc_spear = per_class_corr(a, b, spearmanr)

    print(f"\nOverall flattened Pearson         : {flat_p:.4f}")
    print(f"Per-class Pearson   mean={np.mean(pc_pear):.4f}  median={np.median(pc_pear):.4f}  "
          f"(n_classes={len(pc_pear)})")
    print(f"Per-class Spearman  mean={np.mean(pc_spear):.4f}  median={np.median(pc_spear):.4f}  "
          f"(n_classes={len(pc_spear)})  <-- rank metric, most relevant")

    s = float(np.mean(pc_spear))
    print("\nVERDICT (on mean per-class Spearman):")
    if s > 0.90:
        print(f"  {s:.3f} > 0.90  => HIGHLY correlated. Conclusive even undertrained:")
        print(f"     low ensemble value. Confirms preferring a DIFFERENT family (RegNetY).")
    elif s < 0.85:
        print(f"  {s:.3f} < 0.85  => looks decorrelated, BUT undertrained-smoke => INCONCLUSIVE.")
        print(f"     Could be noise. Would need a fuller run to trust.")
    else:
        print(f"  {s:.3f} in [0.85, 0.90] => moderately correlated; marginal ensemble value.")


if __name__ == "__main__":
    main()
