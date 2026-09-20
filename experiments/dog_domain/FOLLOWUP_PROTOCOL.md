# Frozen-fusion follow-up validation

2026-09-16, before follow-up scores. Motivation: the completed MPDD experiment
showed a small fusion advantage over DINO. Do not tune toward a significant result.
Keep the original validation-selected DINO weight .75 / CLIP weight .25.
Frozen CLIP ViT-B/32 and DINOv3 ViT-B/16, original native preprocessing.
No encoder or projection-head training; no text or Flow efficacy claim.

## MPDD query/gallery selection sensitivity (already-seen identities)

1. Report original paired win/tie/loss counts; exact two-sided paired sign test
   over nonzero identity differences. Zeros are discarded for that test. This
   small-sample test complements, rather than replaces, the paired bootstrap.
2. Use all images from distributed query+gallery (test IDs only). For each of
   100 deterministic repeats (seed 20260916), draw exactly one gallery photo
   per ID and query all remaining photos. Exclude exact decoded duplicates of
   the query from the gallery; record nonevaluable cases.
3. Compute R@1/5/10 and MRR per ID, average across repeats per ID, then bootstrap
   IDs (2,000 resamples) for paired differences. Also report distribution across
   repeats, but do NOT treat 100 repeats as 100 independent datasets. This
   changes gallery size and is not directly comparable with original MPDD scores.

## DogFaceNet new dataset

Use author's aligned 224px archive and provided classes_test/classes_train lists:
https://zenodo.org/records/12578449 (Mougeot, DogFaceNet datasets, version 1).
Verify publisher MD5 plus locally record SHA256, ZIP CRC, RGB image decoding,
folder IDs and split membership. Official classes_test IDs are the primary
external cohort. The entire released aligned dataset is a separately labeled
secondary frozen-encoder cohort: our encoders are not fine-tuned on classes_train,
but pretraining overlap cannot be excluded. This is a custom retrieval protocol,
not reproduction of the original verification protocol or its published scores.

Exclude exact decoded images shared with MPDD development before evaluation and
record them; check all MPDD for overlap separately. Within each DogFaceNet cohort,
deduplicate decoded-identical images within ID deterministically by member name;
exclude all conflicting-ID duplicate groups rather than relabel. Keep only IDs
with >=2 unique images, report exclusions and actual counts.

Primary retrieval: hash-smallest filename per ID (SHA256 of '20260916|member')
is the single gallery image; every remaining photo is a query. Full ID gallery,
no sampled negatives. Macro-average over ID. No gallery choices based on scores.
Secondary gallery-selection sensitivity: 20 deterministic one-photo gallery
draws per ID, query all other images; aggregate per ID before bootstrap.

Systems: CLIP only, DINO only, fixed .25 CLIP + .75 DINO score fusion.
Metrics: R@1/5/10, MRR. Primary contrast: fusion minus DINO on official test IDs,
R@1, two-sided paired ID sign-flip permutation test (100,000 Monte Carlo samples,
seed 20260916; +1 correction), paired ID-bootstrap 95% CI. Other metrics,
gallery repeats, full-cohort and MPDD follow-ups are secondary/exploratory;
no multiplicity-adjusted claims. Report paired ID win/tie/loss counts.

Do not use source image/folder labels to construct text. Do not promote a new
application setting based on this run. New dataset is external to the MPDD
selection, not guaranteed disjoint from foundation-model pretraining or all
unidentified real animals. No raw photo redistribution.
