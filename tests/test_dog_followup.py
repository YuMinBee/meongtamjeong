import numpy as np
import pytest

from experiments.dog_domain.followup import exact_sign_test, gallery_draw, permutation_p, scores_to_identity_metrics


def test_one_additional_success_is_not_significant():
    result = exact_sign_test([0., 0., 1.])
    assert result == {'wins': 1, 'losses': 0, 'ties': 2, 'two_sided_exact_p': 1.}
    assert exact_sign_test([1.] * 6)['two_sided_exact_p'] == .03125
    assert permutation_p([0., 0.], samples=100) == 1.


def test_one_gallery_image_per_identity_and_no_query_overlap():
    records = [{'identity': str(i), 'member': f'{i}/{j}.jpg'} for i in range(5) for j in range(3)]
    query, gallery = gallery_draw(records)
    assert len(query) == 10 and len(gallery) == 5
    assert not set(query) & set(gallery)
    assert {records[i]['identity'] for i in gallery} == {str(i) for i in range(5)}
    np.testing.assert_array_equal(gallery_draw(records)[1], gallery)


def test_ranks_match_full_stable_sort():
    torch = pytest.importorskip('torch')
    matrix = np.array([[.8, .8, .1], [.6, .7, .9]])
    targets = np.array([1, 0])
    macro, rows = scores_to_identity_metrics(torch.tensor(matrix), torch.tensor(targets), targets, 2)
    ranks = [int(np.where(np.argsort(-row, kind='stable') == target)[0][0])+1
             for row, target in zip(matrix, targets)]
    expected = np.array([[float(r <= k) for k in (1,5,10)] + [1/r] for r in ranks])
    np.testing.assert_allclose(rows, expected)
    np.testing.assert_allclose(macro, expected[::-1])
