# DINO fusion application integration

Date: 2026-08-14 (Asia/Seoul)

## Outcome

The pilot is now available to the application through an opt-in runtime
adapter. The default remains the existing CLIP path.

- `DINO_FUSION_MODE=off`: no DINO imports, model load, or ranking change.
- `DINO_FUSION_MODE=shadow`: execute DINO but serve CLIP; expose candidate IDs
  and latency under `retrieval_upgrade`.
- `DINO_FUSION_MODE=active`: use the DINO shared-space score as the vector
  branch of hybrid retrieval.
- `/search/appearance/image`: DINO image plus appearance-only aligned CLIP
  text. Behavior language remains excluded.
- `/recommend_with_image`: the same DINO retrieval plus the public-notice
  behavior branch when a supported preference is explicit.
- `public-text-only-v1`: DINO visual artifacts are always disabled.

The loader verifies the source metadata, DINO index, DINO row metadata, flow
checkpoint and behavior checkpoint hashes. It also checks row counts, vector
dimensions, model ID and resolved DINO revision before serving a score.

## Real runtime smoke

Environment: existing `dog-rag` Conda environment, CUDA, local-files-only,
current 1,002-crop artifacts.

Input:

- one current local dog crop;
- appearance text: `검은색과 흰색 털이 있는 작은 강아지`;
- behavior text: `온순하고 활발한 강아지`.

Observed:

- state: `ready`;
- query mode: `dino_image+aligned_clip_text+behavior`;
- behavior preferences: `gentle_handling=positive`, `activity=positive`;
- mapped candidates: 1,002/1,002;
- cold first request: 3,683.9 ms including model/artifact and behavior-cache
  initialization;
- warm repeat: 39.5 ms total (23.7 ms encode, 15.6 ms search/behavior);
- repeated request returned the same Top-3 order.

With the notice-safety filter disabled **for diagnostic plumbing only**, the
full FastAPI `/recommend_with_image` call returned three candidates and exposed
per-result DINO similarity, DINO percentile, conservative behavior score,
axis-level state and public-notice evidence provenance.

## Current operational blocker

At the actual date of this check, all 1,516 unique notices in the tracked
2026-07-26 snapshot classify as `expired`. The normal safety filters therefore
correctly return zero deployable candidates. This is independent of DINO
ranking and must not be bypassed in a demo or deployment.

Before switching from `shadow` to `active`:

1. build a fresh active-only CLIP metadata/index snapshot;
2. rebuild the DINO index and both learned heads against that exact metadata
   hash;
3. run a new human-authored image+text query set in `shadow` mode;
4. promote only if the predeclared relevance and latency gates pass.

## Interpretation

This is a working application upgrade and rollout mechanism, not yet a
production performance claim. The existing 20-21-query alignment results and
public-notice silver-label behavior results justify continued evaluation, but
they are too small and too indirect to select `active` without fresh data and
human relevance judgments.
