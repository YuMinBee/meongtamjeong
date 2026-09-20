"""Retrieval metrics and paired cluster bootstrap, independent of encoders."""

from __future__ import annotations

import numpy as np


def average_precision(ranking, positives, cutoffs=(5, 10, 25, 50)):
    positives = set(positives)
    if not positives:
        raise ValueError("Average precision needs at least one positive")
    hits = np.array([item in positives for item in ranking], dtype=np.float64)
    precision = hits.cumsum() / np.arange(1, len(hits) + 1)
    return np.array([(hits[:k] * precision[:k]).sum() / min(k, len(positives))
                     for k in cutoffs])


def recall(ranking, positives, cutoffs=(1, 2, 3)):
    positives = set(positives)
    return np.array([float(bool(set(ranking[:k]) & positives)) for k in cutoffs])


def random_average_precision(gallery_size, positive_count, cutoffs=(5, 10, 25, 50)):
    """Exact expectation under a uniformly random permutation, after exclusions."""
    if not 0 < positive_count <= gallery_size:
        raise ValueError("Invalid gallery/positive counts")
    ranks = np.arange(1, min(max(cutoffs), gallery_size) + 1, dtype=np.float64)
    preceding = (ranks - 1) * (positive_count - 1) / max(gallery_size - 1, 1)
    contributions = positive_count / gallery_size * (1 + preceding) / ranks
    return np.array([contributions[:k].sum() / min(k, positive_count) for k in cutoffs])


def stable_topk(scores, k):
    """Exact score-descending/index-ascending top-k, including boundary ties."""
    import torch

    k = min(k, scores.shape[1])
    values, indices = torch.topk(scores, min(k + 1, scores.shape[1]), dim=1, sorted=True)
    if values.shape[1] > k:
        tied_rows = torch.where(values[:, k - 1] == values[:, k])[0].tolist()
        for row in tied_rows:
            # nonzero returns ascending indices, so a stable sort breaks ties by ID.
            candidates = torch.where(scores[row] >= values[row, k - 1])[0]
            order = torch.argsort(scores[row, candidates], descending=True, stable=True)
            indices[row, :k] = candidates[order[:k]]
            values[row, :k] = scores[row, indices[row, :k]]
    values, indices = values[:, :k], indices[:, :k]
    by_id = torch.argsort(indices, dim=1, stable=True)
    indices, values = indices.gather(1, by_id), values.gather(1, by_id)
    by_score = torch.argsort(values, dim=1, descending=True, stable=True)
    return indices.gather(1, by_score)


def cluster_interval(values, groups, *, samples=2000, seed=20260916):
    """Resample clusters; preserve query-weighted means and paired differences.

    Average repeated-seed query scores BEFORE calling. Pass per-query differences
    for a paired CI, rather than subtracting two independently bootstrapped CIs.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if len(values) != len(groups) or not len(values):
        raise ValueError("Values and non-empty groups must align")
    unique, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.zeros((len(unique), values.shape[1]))
    np.add.at(sums, inverse, values)
    rng = np.random.default_rng(seed)
    bootstraps = np.empty((samples, values.shape[1]))
    for i in range(samples):
        sample = rng.integers(0, len(unique), len(unique))
        bootstraps[i] = sums[sample].sum(axis=0) / counts[sample].sum()
    return {"mean": values.mean(axis=0).tolist(),
            "ci95_low": np.quantile(bootstraps, .025, axis=0).tolist(),
            "ci95_high": np.quantile(bootstraps, .975, axis=0).tolist(),
            "clusters": len(unique), "queries": len(values), "resamples": samples}
