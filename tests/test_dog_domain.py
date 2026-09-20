import numpy as np

from experiments.dog_domain.mpdd import identity_means, rank_metrics
from experiments.dog_domain.text_diagnostic import color_ndcg


def test_exact_duplicates_excluded_but_different_photo_positive_retained():
    query = {'identity': 'dog1', 'pixels_sha256': 'same', 'c_code': 'c1', 'member': 'q'}
    gallery = [{**query, 'member': 'duplicate'}, {**query, 'pixels_sha256': 'different', 'c_code': 'c2'},
               {**query, 'identity': 'dog2', 'pixels_sha256': 'negative'}]
    values, _, excluded = rank_metrics(np.array([[1., .5, .7]]), [query], gallery)
    np.testing.assert_allclose(values, [[0, 1, 1, .5, .5]])
    assert excluded == []


def test_identity_macro_not_dominated_by_multiple_queries():
    values, ids = identity_means(np.array([[1.], [1.], [0.]]),
                                [{'identity': 'a'}, {'identity': 'a'}, {'identity': 'b'}])
    assert ids == ['a', 'b']
    assert values.mean() == .5


def test_color_ndcg_excludes_self_and_unknown():
    assert color_ndcg(np.array([1., .9, .8, .7]), ['black', '', 'black', 'white'], 0) == 1.
    assert np.isnan(color_ndcg(np.array([1., .9]), ['black', 'white'], 0))
