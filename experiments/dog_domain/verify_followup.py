"""Independent full-sort audit of the primary external retrieval rankings."""

import numpy as np

from experiments.composed_retrieval.prepare import load_json, write_json
from experiments.dog_domain.followup import DEST


def run():
    records = load_json(DEST / 'dogface_audit.json')['records']
    index = {r['member']: i for i, r in enumerate(records)}
    checked = 0
    rng = np.random.default_rng(20260916)
    with np.load(DEST / 'dogface_features.npz') as features:
        for method in ('CLIP', 'DINO', 'fusion'):
            with np.load(DEST / f'dogface_official_test_{method}_primary.npz') as result:
                gallery = [index[v] for v in result['gallery_members']]
                query_members = result['query_members']
                positions = rng.choice(len(query_members), min(30, len(query_members)), replace=False)
                for position in positions:
                    q = index[query_members[position]]
                    ci = features['clip'][gallery] @ features['clip'][q]
                    di = features['dino'][gallery] @ features['dino'][q]
                    scores = ci if method == 'CLIP' else di if method == 'DINO' else .25 * ci + .75 * di
                    order = np.argsort(-scores, kind='stable')
                    hits = [records[gallery[i]]['identity'] == records[q]['identity'] for i in order]
                    rank = hits.index(True) + 1
                    expected = [float(rank <= k) for k in (1, 5, 10)] + [1 / rank]
                    np.testing.assert_allclose(result['per_query'][position], expected, atol=1e-12)
                    checked += 1
    write_json(DEST / 'independent_verification.json', {'passed': True, 'query_method_checks': checked,
               'method': 'CPU NumPy feature dots + full stable sorting + direct identity matching'})
    print(f'Independent ranking verification passed: {checked} query-method checks')


if __name__ == '__main__':
    run()
