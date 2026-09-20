"""Fixed color-only text controls on an already-inspected external dog corpus."""

import numpy as np

from experiments.composed_retrieval.download import ROOT, sha256
from experiments.composed_retrieval.metrics import cluster_interval
from experiments.composed_retrieval.prepare import load_json, write_json
from experiments.dino_fusion.core import ClipEncoder, normalize_rows
from experiments.dino_fusion_external_eval.evaluate_performance_ablations import COLOR_KO, _load_flow, FLOW_REPORT
from experiments.dog_domain.mpdd import OUT


def color_ndcg(scores, colors, reference):
    eligible = [i for i, color in enumerate(colors) if color and i != reference]
    relevant = np.array([colors[i] == colors[reference] for i in eligible], dtype=float)
    positives = int(relevant.sum())
    if not positives:
        return np.nan
    order = np.argsort(-scores[eligible], kind='stable')[:10]
    dcg = (relevant[order] / np.log2(np.arange(2, len(order) + 2))).sum()
    ideal = (1 / np.log2(np.arange(2, min(10, positives) + 2))).sum()
    return float(dcg / ideal)


def run():
    source = ROOT / 'experiments/dino_fusion_external_eval/artifacts'
    all_records = load_json(source / 'multimodal_records.json')
    records = sorted([r for r in all_records if r['split'] == 'test'], key=lambda r: str(r['pet_id']))
    metadata = load_json(source / 'multimodal_image_embeddings.json')
    if metadata['records_sha256'] != sha256(source / 'multimodal_records.json'):
        raise ValueError('Cached records changed')
    with np.load(source / 'multimodal_image_embeddings.npz') as cache:
        index = {str(v): i for i, v in enumerate(cache['pet_ids'])}
        ids = [index[str(r['pet_id'])] for r in records]
        features = {k: cache[k][ids] for k in ('clip_query', 'clip_gallery', 'dino_query', 'dino_gallery')}
    colors = [r['appearance'].get('color_primary', '') for r in records]
    refs = [i for i, c in enumerate(colors) if c]
    texts = [f"{COLOR_KO.get(colors[i], colors[i])} 털의 강아지를 찾아줘" for i in refs]
    encoder = ClipEncoder(device='cuda')
    torch = encoder._torch
    torch.set_num_threads(4)
    text = encoder.encode_text_batch(texts)
    head = _load_flow('cuda', torch)
    with torch.inference_mode():
        mapped = normalize_rows(head(torch.tensor(text, device='cuda')).cpu().numpy())
    ci, di = features['clip_query'][refs], features['dino_query'][refs]
    cg, dg = features['clip_gallery'], features['dino_gallery']
    scores = {'CLIP_image': ci @ cg.T, 'DINO_image': di @ dg.T,
              'CLIP_text_0.2': normalize_rows(.8 * ci + .2 * text) @ cg.T,
              'late_text_0.2': .8 * (di @ dg.T) + .2 * (text @ cg.T),
              'Flow_text_0.2': normalize_rows(.8 * di + .2 * mapped) @ dg.T,
              'CLIP_shifted_text': normalize_rows(.8 * ci + .2 * np.roll(text, 1, axis=0)) @ cg.T,
              'Flow_shifted_text': normalize_rows(.8 * di + .2 * np.roll(mapped, 1, axis=0)) @ dg.T}
    groups = [records[i]['organization_id'] for i in refs]
    report = {'status': 'completed_exploratory_reuse', 'gallery_dogs': len(records), 'queries': len(refs),
              'text_weight': .2, 'flow_training_report_sha256': sha256(FLOW_REPORT),
              'source_records_sha256': metadata['records_sha256'],
              'image_features_sha256': sha256(source / 'multimodal_image_embeddings.npz'),
              'metric_names': ['R@1', 'R@5', 'R@10', 'MRR', 'known_color_nDCG@10'],
              'shifted_text_same_color_fraction': float(np.mean(np.array(colors)[refs] == np.roll(np.array(colors)[refs], 1))),
              'results': {}}
    arrays = {}
    for method, matrix in scores.items():
        values = []
        for j, ref in enumerate(refs):
            order = np.argsort(-matrix[j], kind='stable')
            rank = int(np.where(order == ref)[0][0]) + 1
            values.append([float(rank <= k) for k in (1, 5, 10)] + [1 / rank, color_ndcg(matrix[j], colors, ref)])
        values = np.array(values)
        arrays[method] = values
        known = np.isfinite(values[:, 4])
        report['results'][method] = {'identity': cluster_interval(values[:, :4], groups),
            'color': cluster_interval(values[known, 4], np.array(groups)[known]),
            'undefined_color_queries': int((~known).sum())}
    pairs = [('Flow_text_0.2', 'DINO_image'), ('Flow_text_0.2', 'CLIP_text_0.2'),
             ('Flow_text_0.2', 'late_text_0.2'), ('Flow_text_0.2', 'Flow_shifted_text'),
             ('CLIP_text_0.2', 'CLIP_shifted_text')]
    report['paired_differences'] = {}
    for first, second in pairs:
        difference = arrays[first] - arrays[second]
        known = np.isfinite(difference[:, 4])
        report['paired_differences'][f'{first} minus {second}'] = {
            'identity': cluster_interval(difference[:, :4], groups),
            'color': cluster_interval(difference[known, 4], np.array(groups)[known])}
    OUT.mkdir(exist_ok=True)
    write_json(OUT / 'text_diagnostic.json', report)
    np.savez_compressed(OUT / 'text_diagnostic_per_query.npz', **arrays,
                        pet_ids=np.array([records[i]['pet_id'] for i in refs]), groups=np.array(groups), texts=np.array(texts))
    lines = ['# Existing dog-corpus text diagnostic', '',
             '**Exploratory reuse of an already-inspected test corpus; not a new independent validation.**', '',
             f"{len(refs)} color-known query dogs, {len(records)} full-photo gallery dogs; text weight fixed at .20.",
             'Original dog-trained Flow; no training or tuning. Query photo and target photo are distinct archive members.',
             'Structured color is proxy relevance. Unknown-color candidates and the query dog are excluded from color nDCG.',
             f"Shifted texts retained the original color for {100*report['shifted_text_same_color_fraction']:.2f}% of queries.", '',
             '| Method | R@1 | R@5 | R@10 | MRR | Color nDCG@10 |', '|---|---:|---:|---:|---:|---:|']
    for method, result in report['results'].items():
        means = result['identity']['mean'] + result['color']['mean']
        lines.append(f'| {method} | ' + ' | '.join(f'{v*100:.2f}' for v in means) + ' |')
    lines.extend(['', '## Paired differences', '', 'Percentage points, organization-cluster bootstrap 95% intervals; no multiplicity correction.', '',
                  '| Contrast | R@1 delta [CI] | Color nDCG delta [CI] |', '|---|---|---|'])
    for pair, result in report['paired_differences'].items():
        cells = [f"{r['mean'][0]*100:+.2f} [{r['ci95_low'][0]*100:+.2f}, {r['ci95_high'][0]*100:+.2f}]" for r in (result['identity'], result['color'])]
        lines.append(f'| {pair} | ' + ' | '.join(cells) + ' |')
    lines.extend(['', 'This diagnostic cannot establish free-form Korean understanding, temperament, deployment benefit, or novel architecture superiority.',
                  'Original photo data has no stated redistribution license in the local dataset card; no raw photos are published here.', ''])
    from pathlib import Path
    Path(__file__).with_name('TEXT_DIAGNOSTIC.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines[:20]), flush=True)


if __name__ == '__main__':
    run()
