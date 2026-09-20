"""Audit saved Taiwan rankings and recompute sampled ranks from embeddings."""
import math
import argparse
from collections import Counter

import numpy as np

from experiments.dog_domain.taiwan_eval import OUT, attributes, read, write, sha256


def run(prefix=""):
    rows = read(OUT / 'records.json')
    report = read(OUT / f'{prefix}results.json')
    labels = [attributes(r) for r in rows]
    counts = Counter(labels)
    assert len({r['animal_id'] for r in rows}) == len(rows)
    assert len({r['pixel_sha256'] for r in rows}) == len(rows)
    checked, sampled = 0, 0
    with np.load(OUT / 'images.npz') as features:
        ci, di = features['clip'], features['dino']
    with np.load(OUT / f'{prefix}top10.npz') as ranks, np.load(OUT / f'{prefix}query_metrics.npz') as values, np.load(OUT / f'{prefix}text_features.npz') as texts:
        for condition, result in report['conditions'].items():
            refs = values[condition + '/refs']
            for method, stats in result['results'].items():
                key = condition + '/' + method
                top, expected = ranks[key], values[key]
                independent = []
                for i, ref in enumerate(refs):
                    assert int(ref) not in top[i]
                    assert len(set(top[i])) == 10
                    truth = [labels[j] == labels[ref] for j in top[i]]
                    positives = counts[labels[ref]] - 1
                    ideal = sum(1 / math.log2(k + 2) for k in range(min(10, positives)))
                    dcg = sum(float(v) / math.log2(k + 2) for k, v in enumerate(truth))
                    independent.append([dcg / ideal, sum(truth) / 10] if ideal else [np.nan, np.nan])
                np.testing.assert_allclose(independent, expected, atol=1e-10, equal_nan=True)
                np.testing.assert_allclose(np.nanmean(independent, axis=0), stats['mean'], atol=1e-10)
                checked += len(refs)
            # Independent single-query float64 scoring for image and native CLIP routes.
            for offset in np.linspace(0, len(refs)-1, min(20, len(refs)), dtype=int):
                ref = refs[offset]
                for method in ['CLIP_image', 'DINO_image', 'CLIP_mix', 'late_mix', 'CLIP_shift', 'CLIP_text_only']:
                    tx = texts[condition][offset].astype(float)
                    if method == 'CLIP_image':
                        score = ci.astype(float) @ ci[ref].astype(float)
                    elif method == 'DINO_image':
                        score = di.astype(float) @ di[ref].astype(float)
                    elif method == 'late_mix':
                        score = .8 * (di.astype(float) @ di[ref].astype(float)) + .2 * (ci.astype(float) @ tx)
                    elif method == 'CLIP_text_only':
                        score = ci.astype(float) @ tx
                    else:
                        if method == 'CLIP_shift':
                            tx = texts[condition][(offset-1) % len(refs)].astype(float)
                        query = .8 * ci[ref].astype(float) + .2 * tx
                        score = ci.astype(float) @ (query / np.linalg.norm(query))
                    score[ref] = -np.inf
                    saved = ranks[condition + '/' + method][offset]
                    threshold = np.sort(score)[-10]
                    assert np.min(score[saved]) >= threshold - 2e-6
                    assert np.all(np.diff(score[saved]) <= 2e-6)
                    sampled += 1
    audit = {'status': 'passed', 'query_method_metrics_checked': checked,
             'independent_float64_rankings_checked': sampled,
             'reference_photo_in_top10': 0, 'results_sha256': sha256(OUT / f'{prefix}results.json')}
    write(OUT / f'{prefix}verification.json', audit)
    print(audit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', action='store_true')
    run('base_' if parser.parse_args().base else '')
