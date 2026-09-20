"""Protect benchmark ground truth and data separation without loading any models."""

import json
from pathlib import Path
import zipfile

from PIL import Image
import pytest

from experiments.composed_retrieval import prepare
from experiments.composed_retrieval.validate import check_query_manifest, inspect_image


def test_official_duplicate_slots_are_preserved_without_positive_first_ties(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "DATA", tmp_path)
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    annotation = tmp_path / "annotations/genecis/focus_attribute.json"
    annotation.parent.mkdir(parents=True)
    reference = {"image_id": "10", "instance_bbox": [0, 0, 12, 12]}
    target = {"image_id": "20", "instance_bbox": [1, 2, 10, 10]}
    annotation.write_text(json.dumps([{"reference": reference, "target": target,
                                       "gallery": [target, reference], "condition": "color"}]))
    manifest = prepare.genecis_manifest("focus_attribute")
    query = manifest["queries"][0]
    assert len(query["candidates"]) == 3
    assert len(set(query["candidates"])) == 3
    assert len(set(query["candidate_assets"])) == 2
    assert len(query["positives"]) == 1
    target_slot = query["candidates"].index(query["positives"][0])
    assert query["candidate_assets"][target_slot] == prepare.asset_for_genecis(target)["key"]
    assert manifest["duplicate_candidate_audit"][0]["target_occurrences"] == 2
    assert check_query_manifest(manifest)["queries"] == 1
    assert prepare.genecis_manifest("focus_attribute") == manifest


def test_missing_positive_and_reference_split_leak_are_rejected():
    query = {"id": "q", "group": "source", "reference": "a", "text": "blue",
             "candidates": ["a", "b"], "positives": ["c"], "split": "audit"}
    manifest = {"assets": [{"key": "a"}, {"key": "b"}], "queries": [query]}
    with pytest.raises(ValueError, match="positive missing"):
        check_query_manifest(manifest)
    query["positives"] = ["b"]
    manifest["queries"].append({**query, "id": "q2", "split": "development"})
    with pytest.raises(ValueError, match="leaks across splits"):
        check_query_manifest(manifest)


def test_training_pool_excludes_all_vg_ids_and_keeps_captions_together(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "DATA", tmp_path)
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.setattr(prepare, "ARTIFACTS", tmp_path / "out")
    monkeypatch.setattr(prepare, "vg_metadata", lambda: [{"coco_id": 0}, {"coco_id": 1}])
    monkeypatch.setattr(prepare, "sha256", lambda _: "fixture-hash")
    jobs = []
    monkeypatch.setattr(prepare, "download_jobs", lambda values, *args, **kwargs: jobs.extend(values))
    captions = [{"image_id": i, "caption": f"caption {i} {variant}"}
                for i in range(6003) for variant in range(2)]
    with zipfile.ZipFile(tmp_path / "annotations_trainval2017.zip", "w") as archive:
        archive.writestr("annotations/captions_train2017.json", json.dumps({"annotations": captions}))
    prepare.prepare_training()
    manifest_path = tmp_path / "out/training_manifest.json"
    manifest = prepare.load_json(manifest_path)
    rows = manifest["records"]
    train = {row["image_id"] for row in rows if row["split"] == "train"}
    validation = {row["image_id"] for row in rows if row["split"] == "validation"}
    assert len(train) == 5000 and len(validation) == 1000
    assert not train & validation
    assert not (train | validation) & {0, 1}
    assert all(len(row["captions"]) == 2 for row in rows)
    assert len(jobs) == 6000
    prepare.prepare_training()
    assert prepare.load_json(manifest_path) == manifest


def test_circo_refuses_a_reduced_gallery(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "DATA", tmp_path)
    with zipfile.ZipFile(tmp_path / "unlabeled2017.zip", "w") as archive:
        archive.writestr("unlabeled2017/000000000001.jpg", b"fixture")
    with pytest.raises(ValueError, match="incomplete CIRCO gallery"):
        prepare.circo_manifest()


def test_circo_split_keeps_transitively_shared_targets_together():
    rows = [{"id": index, "reference_img_id": reference, "gt_img_ids": positives,
             "relative_caption": "different color", "shared_concept": "object",
             "semantic_aspects": ["color"]}
            for index, (reference, positives) in enumerate([(10, [20]), (30, [20, 40]), (50, [40]), (60, [70])])]
    keys = {f"coco:{image_id:012d}" for image_id in (10, 20, 30, 40, 50, 60, 70)}
    queries = prepare.circo_query_records(rows, keys)
    assert len({query["group"] for query in queries[:3]}) == 1
    assert len({query["split"] for query in queries[:3]}) == 1
    assert queries[3]["group"] != queries[0]["group"]
    reverse = prepare.circo_query_records(list(reversed(rows)), keys)
    assert {q["id"]: q["split"] for q in reverse} == {q["id"]: q["split"] for q in queries}
    manifest = {"dataset": "circo", "assets": [{"key": key} for key in keys], "queries": queries}
    assert check_query_manifest(manifest)["independence_groups"] == 2


def test_image_audit_detects_invalid_payload_and_decoded_duplicates(tmp_path, monkeypatch):
    from experiments.composed_retrieval import validate

    monkeypatch.setattr(validate, "ROOT", tmp_path)
    image = Image.new("RGB", (8, 12), (20, 40, 60))
    left, right = tmp_path / "left.png", tmp_path / "right.png"
    image.save(left, compress_level=0)
    image.save(right, compress_level=9)
    first, second = inspect_image(left), inspect_image(right)
    assert first["sha256"] != second["sha256"]
    assert first["decoded_sha256"] == second["decoded_sha256"]
    bad = tmp_path / "bad.jpg"
    bad.write_text("<html>download failed</html>")
    assert "error" in inspect_image(bad)


def test_data_preparation_has_no_model_imports():
    import ast

    folder = Path(prepare.__file__).parent
    for name in ("prepare.py", "download.py", "validate.py", "data_only.py"):
        tree = ast.parse((folder / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            assert not any(module.split(".")[0] in {"torch", "clip", "transformers"} for module in modules)
