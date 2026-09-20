# Data-only preparation audit — 2026-09-16

No GPU embedding extraction or model training has been run by this preparation.
The data-only job completed at 2026-09-16 04:53:33 UTC. Its state is in
`artifacts/data_job.json`; the final deep readiness result is
`artifacts/data_readiness.json` (`ready_for_gpu_experiments: true`, no errors).
GPU experiments were separately authorized afterwards; see the GPU runner and results.

## Completed image and separation checks

| Check | Result |
|---|---:|
| Required Visual Genome originals for both GeneCIS attribute tasks | 23,640 / 23,640 downloaded |
| COCO image-caption alignment subset | 6,000 / 6,000 downloaded |
| Training / checkpoint-validation images | 5,000 / 1,000 |
| Captions attached to those images | 30,017 |
| Individually decoded loose image files | 29,640 |
| Missing loose image files | 0 |
| Decode failures | 0 |
| Train/validation shared source IDs | 0 |
| Alignment pool IDs shared with all Visual Genome-linked COCO IDs | 0 |
| Alignment pool IDs shared with GeneCIS COCO object images | 0 |
| Alignment pool / Visual Genome identical decoded-image hashes | 0 |
| CIRCO complete gallery JPGs | 123,403 / 123,403 |
| Alignment pool / full CIRCO gallery shared source IDs | 0 |

COCO validation ZIP (5,000 images), COCO train/validation annotation ZIP and
Visual Genome metadata ZIP passed CRC and SHA-256 verification. Per-file hashes,
decoded-image hashes and image dimensions are in `artifacts/image_integrity.json`.
These checks do not establish absence of perceptual near-duplicates or foundation
model pretraining overlap.

## GeneCIS original annotation findings

| Task | Queries | Queries with repeated candidates | Queries with target repeated as a distractor |
|---|---:|---:|---:|
| Focus attribute | 2,000 | 6 | 1 |
| Change attribute | 2,112 | 5 | 0 |
| Focus object | 1,960 | 0 | 0 |
| Change object | 1,960 | 1 | 1 |
| Total | 8,032 | 12 | 2 |

The adapter preserves all original candidate slots and their official target
labels. It does not silently remove duplicates or relabel duplicated negatives.
Stable SHA-256 slot IDs define tie order without putting the positive first.
The two contradictory duplicate-target cases are `focus_attribute:14` and
`change_object:19`. Any later clean-subset analysis must be supplementary and
reported alongside the unmodified benchmark, with the exclusion rule declared.

## CIRCO partition audit (no model scores inspected)

The initial reference-ID-only partition shared 5 labeled images across its
development/audit sides, including 4 positive images. The final data-only
partition joins queries through reference AND all annotated positive image IDs.
Connected components are assigned by a deterministic hash.

- Development: 116 queries.
- Held-out local audit: 104 queries.
- Independent labeled-image components: 208.
- Shared reference/positive images across development/audit: **0**.
- CIRCO reference/positive COCO IDs shared with Visual Genome metadata: **0**.

This is a local split of the official 220-query validation annotations, not an
official hidden-test result. Retrieval still uses the complete 123,403-image
gallery. The 20,126,613,414-byte archive finished downloading and passed full
CRC and SHA-256 verification. SHA-256:
`0a57aa17c76037b304e090d6e09c60e55f2f22bbc754d2f5fc692aca4093ef5c`.
The authoritative archive checksum is in the readiness report and `.sha256`
sidecar; image features read members directly without a second extracted copy.

## CIRR availability

Pinned rc2 annotations and image-ID lists are downloaded: 28,225 training,
4,181 validation and 4,148 test queries. Raw images remain pending the official
NLVR2 access procedure; no access form or email has been sent. CIRR is therefore
not ready for new CLIP/DINO encoding.

## Code validation

Seven CPU-only tests pass, including transitive split separation, preserving
official duplicate slots, rejecting incomplete galleries, corrupt payload
detection, decoded duplicate detection and retaining image-level caption splits.
Ruff checks pass for the data preparation code and tests.
