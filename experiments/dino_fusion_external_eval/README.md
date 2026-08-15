# PetFinder external compatibility evaluation

This directory evaluates the experimental DINOv3 + CLIP route on real
photo-plus-text adoption queries. It is isolated from the application runtime;
no checkpoint here is enabled in production.

## Protocol

- Join PetFinder structured behavior fields and descriptions to a separate
  multi-photo archive by PetFinder dog ID.
- Split complete shelter organizations 70/15/15, so one shelter's templates do
  not cross train, validation, and test.
- Use photo 1 as the query and photo 2 as the gallery image.
- Add a Korean query for one known condition: children, dogs, cats, or house
  training.
- Score same-dog Recall@1/5/10 and behavior nDCG@10. Missing compatibility
  fields stay unknown rather than becoming negative.

The joined multimodal slice has 424 dogs from 203 organizations. The untouched
test split has 77 dogs and 203 evaluable image-plus-text queries.

## Reproduce

Prepare the manifests with the lightweight repository environment:

```powershell
python -m experiments.dino_fusion_external_eval.prepare_benchmark
```

Run the frozen baseline, then train the small head and evaluate again:

```powershell
conda run -n dog-rag python -m experiments.dino_fusion_external_eval.evaluate_retrieval
conda run -n dog-rag python -m experiments.dino_fusion_external_eval.train_compatibility
conda run -n dog-rag python -m experiments.dino_fusion_external_eval.evaluate_retrieval
```

The optional evidence-window CLIP control is:

```powershell
conda run -n dog-rag python -m experiments.dino_fusion_external_eval.train_compatibility --text-mode evidence
conda run -n dog-rag python -m experiments.dino_fusion_external_eval.evaluate_retrieval
```

The full-document information-availability control needs scikit-learn:

```powershell
python -m experiments.dino_fusion_external_eval.evaluate_sparse_baseline
```

The independent visual-backbone, crop/full multiview, and Korean appearance
text-weight ablation is:

```powershell
conda run -n dog-rag python `
  -m experiments.dino_fusion_external_eval.evaluate_performance_ablations
```

It selects gallery view and text weight on the organization-disjoint validation
split before applying the frozen choice to test.  See
[`PERFORMANCE_ABLATION_RESULTS.md`](PERFORMANCE_ABLATION_RESULTS.md) for the
measured outcome.

Generated records, embeddings, checkpoints, and reports are kept under the
ignored `artifacts/` directory. Raw image data stays under `tmp/` and must not
be redistributed: the photo dataset card does not state a license.

See [RESULTS.md](RESULTS.md) for the measured outcome and timing.
