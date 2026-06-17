# BirdCLEF+ 2026 — Multi-Teacher Pseudo-Labeling & a Custom EfficientNetV2-s Branch

Kaggle [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026): multi-label
bird-species detection in continuous soundscapes — **234 classes**, scored on 5-second
windows (macro ROC-AUC), under a **90-minute CPU-only, no-internet** inference budget.

**Final result — best private leaderboard submission:**

| | Public LB | Private LB |
|---|---|---|
| `cl2tta_e14_integrated` (3-branch ensemble + my V2-s) | 0.95117 | **0.95224** |

This repo showcases **the part I built** that moved the score, a multi-teacher
pseudo-labeling pipeline, my own EfficientNetV2-s detector trained on it, the
out-of-fold validation harness I used to vet every change, and the ensemble
integration. The Perch embeddings (Google) and the ProtoSSM + distilled-SED base
pipeline are public building blocks I ensembled on top of — they are *not* reproduced here.

## What I contributed

1. **Multi-teacher pseudo-labeling** (`pseudo_labels/`) — generated soft labels for
   unlabeled soundscapes from three teachers (Perch, distilled SED, BirdNET) and
   combined them, expanding usable class coverage to 216/234.
2. **Training my own model** (`training/`) — an **EfficientNetV2-s** SED model trained
   on focal recordings + the expanded pseudo-labels + labeled `train_soundscapes`, with a
   class-balanced sampler and a FocalBCE / SpecAugment / mixup recipe; exported to ONNX
   for fast CPU inference.
3. **Validation methodology** (`validation/`) — a 708×234 out-of-fold soundscape
   harness to measure each change locally (the competition's public LB was a small,
   high-variance sample).
4. **Ensemble integration** (`inference_ensemble/`) — blended the V2-s branch into the
   base pipeline as a third model (**45% ProtoSSM / 30% SED / 25% V2-s**, rank-averaged,
   with temporal-shift TTA).

## Repo layout

```
1_pseudo_labels/      multi-teacher pseudo-label generation + combination
2_training/           train_v2s_combined.py = the model in the winning blend
3_validation/         out-of-fold soundscape validation harness
4_inference_ensemble/ final submission notebook + inference/blend helpers
experiments/          the trail of ablations & dead-ends (see RESULTS.md)
models/               v2s_sed_e14_best.onnx — the trained V2-s (Git LFS, 79 MB)
```

Read it top to bottom: the numbered folders follow the actual workflow
(label → train → validate → serve).

## Notes

- These training scripts ran locally (Python 3.10, 2× RTX 5000 Ada). They read the
  competition audio and a Perch embedding cache; **paths are not parameterized for a
  fresh machine** — this is a portfolio/reference repo, not a one-click reproduction.
- The `.py` scripts here have had inline comments stripped for readability; module
  docstrings (which describe each script's recipe) are kept.
- Heavy artifacts (checkpoints, OOF arrays, caches, intermediate CSVs) are intentionally
  excluded — see `.gitignore`.

## Stack

PyTorch · timm (EfficientNetV2-s, EfficientNet-B0) · ONNX Runtime · TensorFlow (Perch) ·
librosa · LightGBM / scikit-learn (probes) · NumPy / pandas. See `requirements.txt`.
