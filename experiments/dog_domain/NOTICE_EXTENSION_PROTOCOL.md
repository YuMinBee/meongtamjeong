# New Korean notice evaluation, 2026-09-16

Frozen before new scores are computed. This is an extension of an already explored
research hypothesis, not an externally preregistered study.

- Source: official Korean national animal protection API, 32-day window ending
  2026-09-16; include closed notices for offline research. Raw snapshot is local
  under `tmp/notice_extension_20260916/`. Never update the app index.
- Exclude notice IDs in all available old dog metadata, August refresh metadata,
  and the original held-out report. Require happen date >= 20260816. This checks
  notice overlap; distinct notice IDs do not guarantee distinct biological dogs.
- Require known normalized public color and weight-derived size. Choose at most
  1,000 notices by SHA256 of `notice-extension-v1:<noticeID>`, before scoring.
  Download each selected notice's primary and alternate photos, at most four.
  Missing/duplicate photos are exclusions, never replace with another notice.
- Compare full photos for BOTH encoders, primary as gallery, first sufficiently
  different alternate as query. Exact RGB duplicates and conservative near-copy
  pairs (64-bit dHash distance <=3 AND resized RGB MAE < .035) are rejected.
  Remove any notice sharing an exact RGB image with another selected notice.
  This is not the same crop protocol as the original 20-query pilot.
- Frozen CLIP ViT-B/32 and DINOv3 ViT-B/16, revision
  `5931719e67bbdb9737e363e781fb0c67687896bc`; original August Linear/MLP/Flow
  heads verified against their training report. No retraining, selection, or
  tuning on this new test. Fixed image .8 + text .2 throughout.
- Compare image-only CLIP/DINO, CLIP image+text, DINO-image/CLIP-text late scoring,
  and DINO image + aligned Linear/MLP/Flow text. Evaluate Korean and English
  metadata templates; cyclically shifted text controls for CLIP and Flow.
  Evaluate original Korean public `specialMark` descriptions as an additional
  identity-retrieval diagnostic (77-token CLIP truncation recorded).
- Primary semantic proxy: binary color intersection AND size match nDCG@10,
  excluding the query notice itself. All candidates have known color/size.
  Secondary identity proxy: same-notice R@1/5/10 and MRR using distinct photos.
  A notice may contain more than one animal; neither metric is verified re-ID.
- Report all methods and shelter-cluster paired bootstrap 95% intervals (2,000
  resamples). Primary contrasts: Flow vs CLIP image+text and Flow vs DINO image.
  Other comparisons are exploratory, without multiplicity correction.
- Template scores cannot establish free-form understanding. Public description
  identity scores also cannot establish that every phrase was understood.
  No human-relevance labels, temperament claims, raw-photo redistribution, or
  automatic claims of novel architecture superiority.
