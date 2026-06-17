import numpy as np
from scipy.stats import rankdata


def per_class_ranks(scores):
    r = np.zeros_like(scores, dtype=np.float32)
    n = scores.shape[0]
    for c in range(scores.shape[1]):
        r[:, c] = (rankdata(scores[:, c], method="average") - 1) / max(1, n - 1)
    return r


def quantile_mix(scores, alpha=0.5):
    return alpha * scores + (1 - alpha) * per_class_ranks(scores)


def rank_mean_ensemble(list_of_scores, weights=None):
    if weights is None:
        weights = [1.0] * len(list_of_scores)
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / weights.sum()
    out = np.zeros_like(list_of_scores[0], dtype=np.float32)
    for s, w in zip(list_of_scores, weights):
        out += w * per_class_ranks(s)
    return out


def hybrid_ensemble(list_of_scores, weights=None, alpha=0.5):
    if weights is None:
        weights = [1.0] * len(list_of_scores)
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / weights.sum()
    out = np.zeros_like(list_of_scores[0], dtype=np.float32)
    for s, w in zip(list_of_scores, weights):
        out += w * quantile_mix(s, alpha=alpha)
    return out
