# Taiwan official adoption notice snapshot

Source: [Taiwan Ministry of Agriculture — animal adoption](https://data.gov.tw/dataset/85903).
Snapshot retrieved 2026-09-16 under Taiwan Open Government Data License 1.0.
Official JSON endpoint and snapshot SHA256 are retained in `source.json`.

Local data directory: `tmp/notice_extension_20260916/taiwan/` (git ignored).

Download completed 2026-09-16, including one retry pass:

| Item | Count |
|---|---:|
| Dog notices | 5,933 |
| Notices with photo URLs attempted | 5,516 |
| Photos successfully stored | 5,486 |
| Remaining failures | 30 |
| Unique decoded RGB photos | 5,475 |
| Unique photos with nonempty color and size | 5,427 |
| Unique photos with nonempty original remarks | 1,846 |
| Shelters among unique-photo records | 33 |

Storage: 9,765,028,074 bytes (about 9.77 GB). Eleven identical-image pairs were
grouped; raw records remain intact. Remaining failures: 15 DNS failures, seven
timeouts, seven HTTP 406 responses, and one response over the configured size
limit. The 417 notices with no photo URL were not downloaded. Nonempty fields
have not yet been normalized or validated as evaluation labels.

- `notices.json`: unchanged collected metadata for all 8,341 animals.
- `source.json`: provenance; 5,933 dog notices, of which 5,516 have photo URLs.
- `images/<animal_id>.png`: successfully decoded photographs, saved losslessly
  as RGB PNG. Original compressed download bytes are hashed, not retained.
- `downloads/<animal_id>.json`: resumable per-photo status, timestamp, source URL,
  dimensions, source payload hash, stored-file hash, and decoded-pixel hash.
- `image_records.json`: successful photos joined to original color, size, age,
  breed, shelter ID, and original description/caption. No generated labels or translations.
- `unique_image_records.json`: one deterministic representative per identical
  decoded RGB image. Choose representatives by string-sorted animal ID.
- `exact_duplicate_groups.json`: all notice IDs sharing identical decoded RGB pixels.
- `failed_downloads.json`: explicit failures; missing files are never replaced by synthetic photos.
- `download_report.json`: final counts, coverage, file hashes, and storage size.

## Reproduce or resume

Run in the project's `dog-rag` environment from the repository root:

```powershell
python -m experiments.dog_domain.download_taiwan
python -m experiments.dog_domain.download_taiwan --retry-failures
```

Successful cached files are hash-checked and reused. The downloader uses eight
workers with bounded time, payload, pixel count and permitted source hosts.
The metadata snapshot hash is checked before every run.

## Evaluation use

This download does not run training or evaluation. Keep the new corpus unscored
until its evaluation protocol is fixed. A single photo per notice is sufficient
for text-to-image retrieval or photo-plus-text attribute retrieval **with the
query's own photo and its duplicate group excluded from the gallery**. It cannot
support meaningful same-animal cross-photo retrieval without another photo.

Public color and size are shelter annotations, not independent human relevance
judgments. Taiwan's categorical size must not silently be treated as identical
to the Korean weight-derived size bins. Missing descriptions stay missing.
Exact-image de-duplication does not detect all resized/edited copies or prove
that photographs contain only one animal; those checks remain for benchmark design.

Raw photos and contact-bearing source metadata remain local. Application indexes
and model checkpoints are not modified by this downloader.
