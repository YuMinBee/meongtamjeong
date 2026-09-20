# Fixed-photo condition-change evaluation

Specified 2026-09-17 before computing these new outcomes. This is an exploratory
extension of previously inspected datasets, not an untouched external test or
preregistered confirmatory experiment.

- Reuse frozen Korean (611) and Taiwan (5,427) image/text caches and original
  Linear/MLP/Flow checkpoints. CPU only; no downloads, training or weight tuning.
- Run English and Korean separately in both datasets. Keep image/text weights
  .8/.2. Reuse exact existing template embeddings, not new model-generated text.
- For each photo, create at most one size-change and one color-change request.
  Size change preserves the exact color set; color change preserves size and
  uses a disjoint color set. Korean template age is held constant in both cases.
  Choose among existing template signatures by seeded SHA256, independently of
  model scores. Require at least 10 relevant candidates. Record exclusions.
- Korean relevance follows the original color intersection AND size match;
  Taiwan follows exact color-set AND size match. Age is held constant in the
  request but is NOT a scored attribute. Never pool dataset absolute scores.
- Exclude the reference notice/photo from every ranking. All conditions remain
  silver metadata labels, not verified appearance, temperament or preference.
- Compare CLIP image, DINO image, CLIP image+text, unaligned late score fusion,
  aligned Linear/MLP/Flow image+text, and each respective text-only method.
  Add metadata-filter+DINO as a privileged-information diagnostic: filter
  using the exact labels used by the scorer. It is not a fair image-only encoder
  baseline, nor evidence of superior preference recommendation.
- Primary metrics: target-condition nDCG@10 and P@10 after the changed request;
  paired target P@10 gain over the SAME photo with its ORIGINAL text. Image-only
  methods must have exactly zero gain. Report whether the top-10 set changed.
- Report each axis/language separately; resample reference shelters (2,000
  paired cluster bootstraps), preserving repeated queries inside each shelter.
  Compare Flow against CLIP mix, DINO image, Flow text-only and MLP mix.
  Intervals are exploratory and not multiplicity-adjusted. No architecture
  superiority claim from one favorable condition or cherry-picked weight.
- Independent verification: recompute every metric from saved rankings and
  labels; independently recalculate a deterministic sample of rankings in
  float64 from cached features/checkpoints. Preserve manifests and source hashes.
- These tests measure response to attribute changes, not retained photographic
  style or user satisfaction. The reference photographs do not change: there is
  no claim that photo input improves performance unless a relevant independent
  visual/preference evaluation demonstrates it.

Artifacts: D:/autonomous_ai_challenge/archive is unrelated. New dog-search
artifacts go to D:/meongtamjeong_research/condition_change_20260917.
