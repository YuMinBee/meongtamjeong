"""Holm-adjusted bootstrap p-values for the pre-specified comparisons; see STRONG_CLIP_PROTOCOL.md.

CPU only; reads the per-query arrays written by related_baselines, petfinder_identity and strong_clip.
"""
from pathlib import Path
import numpy as np
from experiments.dog_domain import related_baselines as rb

RESULTS = Path(__file__).with_name('MULTIPLICITY_RESULTS.md')
SAMPLES, SEED = 2000, 20260916  # same resampling as composed_retrieval.metrics.cluster_interval


def bootstrap_p(values, groups):
    """Two-sided p from the cluster bootstrap distribution of the mean paired difference."""
    values = np.asarray(values, dtype=np.float64)
    unique, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=values)
    rng = np.random.default_rng(SEED)
    means = np.empty(SAMPLES)
    for i in range(SAMPLES):
        sample = rng.integers(0, len(unique), len(unique))
        means[i] = sums[sample].sum() / counts[sample].sum()
    # (k + 1) / (B + 1) keeps p strictly positive with a finite number of resamples.
    low, high = ((means <= 0).sum() + 1) / (SAMPLES + 1), ((means >= 0).sum() + 1) / (SAMPLES + 1)
    return float(values.mean()), min(1.0, 2 * min(low, high))


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def families():
    out = {}
    rows = []
    for name in rb.DATASETS:
        f = np.load(rb.OUT / f'{name}_per_query.npz')
        known, groups = f['known'], f['groups']
        for a, b in [('dinoL_linear_mix', 'dinotxt_mix'), ('dino_linear_mix', 't2d_mix'), ('dinotxt_mix', 'dinotxt_image'), ('t2d_mix', 't2d_image'),
                     ('DINOL_image', 'dinotxt_image'), ('dinoL_linear_mix', 'DINOL_image'), ('t2darch_mix', 't2d_mix'), ('dinoL_mlp_mix', 'dinotxt_mix'),
                     ('dinotxt_mix', 'CLIP_mix'), ('t2d_mix', 'CLIP_mix'), ('dinoL_linear_mix', 'dino_linear_mix'), ('t2darch_mix', 'dino_linear_mix')]:
            rows.append((f'{name} nDCG@10', f'{a} − {b}', *bootstrap_p((f[a] - f[b])[known, 0], groups[known])))
        if name == 'Korea':
            for a, b in [('DINOL_image', 'dinotxt_image'), ('dinotxt_mix', 'dinotxt_image'), ('t2d_mix', 't2d_image')]:
                rows.append(('Korea R@1', f'{a} − {b}', *bootstrap_p(f[f'identity_{a}'][:, 0] - f[f'identity_{b}'][:, 0], groups)))
    out['dino.txt · Talk2DINO 기준선 (RELATED_BASELINES)'] = rows

    f = np.load(rb.OUT / 'petfinder_identity/per_query.npz')
    keep, groups = ~f['near_copy'], f['groups']
    out['PetFinder 같은 개 찾기 (PETFINDER_IDENTITY)'] = [
        ('PetFinder R@1', f'{a} − {b}', *bootstrap_p((f[a] - f[b])[keep, 0], groups[keep]))
        for a, b in [('DINO_image', 'CLIP_image'), ('dino_linear_mix', 'CLIP_mix'), ('dino_linear_mix', 'DINO_image'),
                     ('DINOL_image', 'dinotxt_image'), ('dinoL_linear_mix', 'dinotxt_mix'), ('dinoL_linear_mix', 'DINOL_image'),
                     ('dinotxt_mix', 'dinotxt_image'), ('t2d_mix', 't2d_image')]]

    rows = []
    for name in rb.DATASETS:
        f = np.load(rb.OUT / f'strong_clip/{name}_per_query.npz')
        known, groups = f['known'], f['groups']
        for a, b in [('dino_linear_mix', 'siglip2L_mix'), ('dinoL_linear_mix', 'siglip2L_mix'), ('dino_linear_mix', 'siglip2B_mix'), ('dino_linear_mix', 'clipL_mix')]:
            rows.append((f'{name} nDCG@10', f'{a} − {b}', *bootstrap_p((f[a] - f[b])[known, 0], groups[known])))
        if name == 'Korea':
            for a, b in [('DINO_image', 'siglip2L_image'), ('DINOL_image', 'siglip2L_image')]:
                rows.append(('Korea R@1', f'{a} − {b}', *bootstrap_p(f[f'identity_{a}'][:, 0] - f[f'identity_{b}'][:, 0], groups)))
    f = np.load(rb.OUT / 'strong_clip/PetFinder_identity_per_query.npz')
    keep, groups = f['keep'], f['groups']
    for a, b in [('DINO_image', 'siglip2L_image'), ('DINOL_image', 'siglip2L_image')]:
        rows.append(('PetFinder R@1', f'{a} − {b}', *bootstrap_p((f[a] - f[b])[keep, 0], groups[keep])))
    out['강한 CLIP 계열 기준선 (STRONG_CLIP)'] = rows
    return out


def main():
    result, lines = {}, ['# 사전 지정 비교의 다중비교 보정 (Holm)', '',
                          '[프로토콜](STRONG_CLIP_PROTOCOL.md#다중비교-보정). 문서별로 한 묶음. p는 보호소/등록자 단위 paired bootstrap 2,000회에서 계산한 양측 값이며 '
                          '(k+1)/(B+1) 방식이라 최소값이 약 0.001이다. 차이는 × 100.', '']
    for family, rows in families().items():
        adjusted = holm(np.array([r[3] for r in rows]))
        result[family] = [dict(metric=m, comparison=c, difference=d, p=p, holm=h) for (m, c, d, p), h in zip(rows, adjusted)]
        lines += [f'## {family}', '', f'비교 {len(rows)}개. Holm 보정 후 5%에서 유의: {int((adjusted < .05).sum())}개.', '',
                  '| 지표 | 비교 | 차이 | p | Holm p | 5% 유의 |', '|---|---|---:|---:|---:|:---:|']
        for (m, c, d, p), h in zip(rows, adjusted):
            lines.append(f'| {m} | {c} | {d * 100:+.2f} | {p:.4f} | {h:.4f} | {"✓" if h < .05 else ""} |')
        lines.append('')
    rb.write('multiplicity.json', result)
    RESULTS.write_text('\n'.join(lines), encoding='utf-8')
    print('MULTIPLICITY COMPLETE', flush=True)


if __name__ == '__main__':
    main()
