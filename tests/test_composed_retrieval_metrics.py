"""Hand-computed retrieval and loss checks for paper experiments."""

import numpy as np
import pytest

from experiments.composed_retrieval.metrics import average_precision, cluster_interval, recall, stable_topk


def test_ap_denominator_and_multiple_positives():
    values = average_precision([4, 1, 3, 2], {1, 2, 7}, cutoffs=(1, 2, 4))
    np.testing.assert_allclose(values, [0., .25, 1. / 3.])
    np.testing.assert_array_equal(recall([4, 1, 3], {1}), [0, 1, 1])
    with pytest.raises(ValueError):
        average_precision([1, 2], set())


def test_random_ap_matches_exhaustive_permutations():
    from itertools import permutations
    from experiments.composed_retrieval.metrics import random_average_precision

    cutoffs = (1, 2, 3, 4)
    empirical = np.mean([average_precision(row, {0, 1}, cutoffs) for row in permutations(range(4))], axis=0)
    np.testing.assert_allclose(random_average_precision(4, 2, cutoffs), empirical)


def test_topk_exact_ties_including_boundary():
    torch = pytest.importorskip("torch")
    scores = torch.tensor([[1., 2., 2., 2., 2.], [1., 1., 1., 1., 1.],
                           [3., 2., 4., 0., -float("inf")]])
    expected = torch.argsort(scores, dim=1, descending=True, stable=True)
    for k in (1, 2, 3, 5):
        assert torch.equal(stable_topk(scores, k), expected[:, :k])
    generator = torch.Generator().manual_seed(7)
    random_scores = torch.randint(0, 4, (100, 50), generator=generator).float()
    expected = torch.argsort(random_scores, dim=1, descending=True, stable=True)[:, :10]
    assert torch.equal(stable_topk(random_scores, 10), expected)


def test_duplicate_images_are_not_false_negatives():
    torch = pytest.importorskip("torch")
    from experiments.composed_retrieval.train import multi_positive_loss

    embeddings = torch.eye(2)
    ids = torch.tensor([0, 1])
    original = multi_positive_loss(embeddings, embeddings, ids, ids, temperature=1.)
    repeated = multi_positive_loss(embeddings, embeddings.repeat_interleave(2, dim=0),
                                   ids, ids.repeat_interleave(2), temperature=1.)
    torch.testing.assert_close(original, repeated)
    with pytest.raises(ValueError, match="positive"):
        multi_positive_loss(embeddings, embeddings, ids + 4, ids)


def test_bootstrap_pairs_seeds_not_independent_queries():
    values = np.array([[1., 2.], [1., 2.], [1., 2.]])
    result = cluster_interval(values, ["a", "a", "b"], samples=100)
    assert result["clusters"] == 2
    assert result["queries"] == 3
    assert result["mean"] == result["ci95_low"] == result["ci95_high"] == [1., 2.]
    paired = cluster_interval(values - values, ["a", "a", "b"], samples=100)
    assert paired["ci95_low"] == paired["ci95_high"] == [0., 0.]


def test_genecis_crop_matches_published_asymmetric_dilation(tmp_path, monkeypatch):
    from PIL import Image
    from experiments.composed_retrieval import features

    monkeypatch.setattr(features, "ROOT", tmp_path)
    pixels = np.arange(50 * 50 * 3, dtype=np.uint8).reshape(50, 50, 3)
    source = Image.fromarray(pixels)
    source.save(tmp_path / "source.png")
    image, byte_hash, pixel_hash = features.image_for_asset(
        {"path": "source.png", "bbox": [20, 15, 10, 20]})
    # left=13, top=1, right=30, bottom=35 => 17x34, centered at x=8.
    expected = Image.new("RGB", (34, 34))
    expected.paste(source.crop((13, 1, 30, 35)), (8, 0))
    np.testing.assert_array_equal(image, expected)
    assert len(byte_hash) == len(pixel_hash) == 64


def test_score_composition_matches_direct_vector_cosine():
    torch = pytest.importorskip("torch")
    from experiments.composed_retrieval.evaluate import Scorer

    scorer = object.__new__(Scorer)
    scorer.torch = torch
    image = torch.nn.functional.normalize(torch.tensor([[1., 2., 3.], [-1., 2., 1.]]), dim=1)
    text = torch.nn.functional.normalize(torch.tensor([[3., 2., 1.], [1., 1., 2.]]), dim=1)
    gallery = torch.eye(3)
    for weight in (0., .2, .5, 1.):
        actual = scorer.composition(image @ gallery.T, text @ gallery.T, image, text, weight)
        expected = torch.nn.functional.normalize((1 - weight) * image + weight * text, dim=1) @ gallery.T
        torch.testing.assert_close(actual, expected)


def test_cache_identity_ignores_only_callable_memory_address():
    from experiments.composed_retrieval.features import canonical_provenance, digest

    first = {"clip_preprocess": "<function rgb at 0x00000123>", "weights": "abc"}
    second = {"clip_preprocess": "<function rgb at 0xFFFF1111>", "weights": "abc"}
    assert digest(canonical_provenance(first)) == digest(canonical_provenance(second))
    second["weights"] = "changed"
    assert digest(canonical_provenance(first)) != digest(canonical_provenance(second))


def test_actual_scorer_excludes_circo_reference_and_preserves_genecis_slots():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Scorer uses CUDA for the full retrieval gallery")
    from experiments.composed_retrieval.evaluate import Scorer

    index = {"image_keys": ["reference", "target", "negative"], "texts": ["condition"]}
    vectors = np.array([[1., 0.], [.8, .6], [0., 1.]], dtype=np.float32)
    features = {"clip": vectors, "dino": vectors, "text": vectors[1:2]}
    query = {"id": "q", "reference": "reference", "text": "condition", "split": "audit",
             "positives": ["target"]}
    manifest = {"dataset": "circo", "assets": [{"key": key} for key in index["image_keys"]],
                "queries": [query]}
    scorer = Scorer(manifest, index, features, {}, split="audit")
    np.testing.assert_array_equal(scorer.score({"family": "clip_image"}), [[1., 1., 1., 1.]])
    query = {**query, "split": "external", "candidates": ["a", "b", "c"],
             "candidate_assets": ["target", "target", "negative"], "positives": ["b"]}
    manifest = {**manifest, "dataset": "genecis_fixture", "queries": [query]}
    scorer = Scorer(manifest, index, features, {}, split="external")
    np.testing.assert_array_equal(scorer.score({"family": "clip_text"}), [[0., 1., 1.]])
