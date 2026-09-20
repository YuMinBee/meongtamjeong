# Taiwan translated-text retrieval protocol — 2026-09-16

Fixed before Taiwan model scores. Extension of an explored hypothesis, not an
external preregistration. Use all downloaded unique dog photos with known color
and size; no training, weight selection, or selection by performance.

- Reference photo + text retrieves OTHER notices. Exclude the reference itself;
  do not report same-animal identity accuracy from a single photo.
- Both encoders use the same full photos. Remove near-copy components before
  scoring: dHash distance <=3 and 32x32 RGB mean absolute difference /255 < .035.
  Retain the lexicographically first animal ID in each component. Exact copies
  were already removed by the downloader. This does not detect all edited copies.
- Translate structured Chinese color and size values using a fixed bilingual
  dictionary. English and Korean templates are controlled attribute queries,
  not translations of original prose. Taiwan size categories retain their own
  identity; do not equate them to Korean weight bins.
- Separately translate all nonempty original remarks to English locally with
  `facebook/nllb-200-distilled-600M` (pinned resolved revision, zho_Hant -> eng_Latn).
  Preserve raw text and translations. Split on sentence punctuation/newlines;
  split overlong segments into <=400 source tokens. Deterministic beam size 4,
  no sampling, max 512 generated tokens per segment. Record capped outputs.
  Never pass images or structured labels to the translator. No translation
  choice based on retrieval scores. No human-validated translation claim.
- CLIP ViT-B/32; DINOv3 ViT-B/16 revision
  `5931719e67bbdb9737e363e781fb0c67687896bc`; original Korean-trained Linear,
  MLP and Flow checkpoints verified against their training report. Fixed text
  weight .20. Compare image-only, CLIP image+text, DINO-image/CLIP-text late
  scores, aligned heads, cyclic shifted texts, and text-only retrieval.
- Primary relevance: same full normalized color-token set AND same Taiwan size.
  Metric: nDCG@10 (binary gains). Also report precision@10. Exclude queries with
  no other relevant candidate and report their count. No image/self-ID metric.
- Primary contrasts: Korean/English template Flow vs CLIP mix and vs DINO image.
  Report paired shelter-cluster bootstrap 95% intervals (2,000 samples). Other
  comparisons, including original/translated prose, are exploratory and are not
  multiplicity-adjusted. Compare Chinese original and English translation on the
  SAME nonempty-description query subset and full common gallery.
- Prose scores still use color/size proxy relevance; they are NOT judgments of
  every sentence's meaning. Assess translator samples and token truncation as
  diagnostics. Full prose may discuss adoption logistics instead of appearance.
- CLIP 77-token truncation is counted per condition. Keep all methods/results,
  including negative results. Raw data and translations remain local and ignored.
