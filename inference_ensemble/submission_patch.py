
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/kaggle/input/birdclef2026-upgrade-utils")
from inference_cnn import infer_cnn

assert "filename" in meta_test.columns, "meta_test needs filename column"
if "window_id" not in meta_test.columns:
    meta_test = meta_test.copy()
    meta_test["window_id"] = meta_test.groupby("filename").cumcount()

print("Running CNN inference on test soundscapes...")
scores_test_cnn = infer_cnn(
    test_paths=test_paths,
    meta_test=meta_test,
    n_classes=N_CLASSES,
    tta_shifts_sec=(0.0,),
)
print("CNN scores shape:", scores_test_cnn.shape)


from ensemble_utils import hybrid_ensemble


CNN_WEIGHT  = 0.45
PROTO_WEIGHT = 0.55

final_scores = hybrid_ensemble(
    [scores_test_proto_mlp, scores_test_cnn],
    weights=[PROTO_WEIGHT, CNN_WEIGHT],
    alpha=0.5,
)


row_ids = [f"{fn}_{5*(w+1)}" for fn, w in zip(meta_test["filename"], meta_test["window_id"])]
sub = pd.DataFrame(final_scores, columns=PRIMARY_LABELS)
sub.insert(0, "row_id", row_ids)
sub.to_csv("submission.csv", index=False)
print("submission.csv written:", sub.shape)

