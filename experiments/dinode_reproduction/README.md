# Official DINOde reproduction

This directory reproduces the official released DINOde checkpoint on the
VOC20 open-vocabulary semantic-segmentation protocol. It is intentionally
separate from `experiments/dino_fusion`: the latter is our dog-retrieval
pilot, while this run tests the authors' actual method and benchmark.

## Fixed target

- Upstream: <https://github.com/yoon307/DINOde.git>
- Commit: `0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1`
- Protocol: VOC20 (background excluded)
- Inference: 448-pixel sliding window, stride 224, no DINO feature cache
- Official expected result: **91.1495 mIoU** (`--no_cache`)
- Vision/text towers: DINOv3 ViT-L/16 and OpenCLIP ViT-L/14

On Windows, the harness generates a runtime copy of the official config with
`data.num_workers=0`. The upstream `collate_val` is a local function and cannot
be pickled by Windows spawn workers. This affects data-loading throughput only;
model operations and evaluation settings are unchanged.

The pinned upstream commit contains no explicit license file. For that
reason, `vendor/` is ignored and the upstream source is not redistributed by
this experiment. Clone it locally at the exact commit:

```powershell
git clone https://github.com/yoon307/DINOde.git vendor/DINOde
git -C vendor/DINOde checkout 0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1
```

## Run

From this directory in PowerShell:

```powershell
.\setup_env.ps1
.\prepare_voc20.ps1
.\run_voc20.ps1
```

For a one-image smoke test before the full 1,449-image evaluation:

```powershell
.\run_voc20.ps1 -MaxSamples 1 -OutputName smoke_voc20
```

The official dependency versions are installed in the independent
`dinode-repro` Conda environment. Raw VOC data, the 205 MB DINOde checkpoint,
Hugging Face/OpenCLIP weights, and generated output are ignored. The official
evaluator prefixes output directories with a timestamp, so the final log is
written under
`vendor/DINOde/outputs/<timestamp>_eval_voc20_no_cache/eval_log.txt`.

This first stage evaluates the released checkpoint; it does **not** retrain
DINOde. Full official training needs COCO-Stuff plus roughly 270 GB when
features are cached, so training is a separate second-stage decision after the
checkpoint result is reproduced.

The completed local result and interpretation are recorded in
[`RESULTS.md`](RESULTS.md): VOC20 reproduced exactly at **91.1495 mIoU**.
