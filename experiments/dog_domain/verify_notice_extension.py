"""Independent CPU ranking audit of the new-notice experiment artifacts."""
import hashlib
import math

import numpy as np

from experiments.dog_domain.notice_extension import OUT, PROTOCOL, old_ids, read, write
from experiments.composed_retrieval.download import sha256


def run():
    records = read(OUT / 'records.json')
    selection = read(OUT / 'selection.json')
    report = read(OUT / 'results.json')
    ids = [r['notice_id'] for r in records]
    assert len(ids) == len(set(ids)) == report['n']
    previous, _ = old_ids()
    assert not set(ids) & previous
    assert set(ids) <= set(selection['selected_ids'])
    assert all(str(r['happen_date']).replace('-', '') >= '20260816' for r in records)
    hashes = [h for r in records for h in r['pixel_hashes']]
    assert len(set(hashes)) == len(hashes)
    assert selection['protocol_sha256'] == sha256(PROTOCOL)
    with np.load(OUT / 'features.npz') as f:
        for name in f.files:
            if name != 'notice_ids':
                np.testing.assert_allclose(np.linalg.norm(f[name], axis=1), 1, atol=1e-5)
        cq, cg, dq, dg, text = [f[k] for k in ['clip_query','clip_gallery','dino_query','dino_gallery','text_korean']]
    checked = 0
    with np.load(OUT / 'scores.npz') as scores, np.load(OUT / 'per_query.npz') as values:
        np.testing.assert_allclose(scores['DINO_image'], dq @ dg.T, atol=1e-6)
        mix = .8 * cq + .2 * text
        mix /= np.linalg.norm(mix, axis=1, keepdims=True)
        np.testing.assert_allclose(scores['korean/CLIP_mix'], mix @ cg.T, atol=1e-6)
        for method in report['results']:
            matrix = scores[method]
            computed = []
            for i, row in enumerate(matrix):
                ranked = sorted(range(len(row)), key=lambda j: (-float(row[j]), j))
                rank = ranked.index(i) + 1
                attr = records[i]['attributes']
                relevant = {j for j, r in enumerate(records) if j != i and
                            set(attr['colors']).intersection(r['attributes']['colors']) and attr['size'] == r['attributes']['size']}
                retrieved = [j for j in ranked if j != i][:10]
                dcg = sum(1 / math.log2(k+2) for k, j in enumerate(retrieved) if j in relevant)
                ideal = sum(1 / math.log2(k+2) for k in range(min(10, len(relevant))))
                computed.append([float(rank <= 1), float(rank <= 5), float(rank <= 10), 1/rank, dcg/ideal if ideal else np.nan])
            np.testing.assert_allclose(computed, values[method], atol=1e-10, equal_nan=True)
            np.testing.assert_allclose(np.mean(np.array(computed)[:, :4], axis=0), report['results'][method]['identity']['mean'])
            np.testing.assert_allclose(np.nanmean(np.array(computed)[:, 4]), report['results'][method]['semantic']['mean'][0])
            checked += len(computed)
    summary = {'status': 'passed', 'independent_query_method_checks': checked,
               'new_notice_ids': len(ids), 'cross_notice_exact_pixel_duplicates': 0,
               'old_notice_overlap': 0, 'protocol_sha256': sha256(PROTOCOL),
               'results_sha256': sha256(OUT / 'results.json'),
               'notice_ids_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest()}
    write(OUT / 'verification.json', summary)
    print(summary)


if __name__ == '__main__':
    run()
