# Taiwan translated-text benchmark

This experiment evaluates retrieval of other adoption notices using a reference
photo and an appearance query or original notice prose. It is separate from
same-animal cross-photo retrieval and does not report identity accuracy.

Data: 5,427 downloaded, unique photos with interpretable public color and Taiwan
size categories, 33 shelters. Five notices have no other matching color+size
candidate and are excluded from template metric means (5,422 evaluable queries).
The original-remarks subset contains 1,824 notices. All methods share the same
gallery and exclude the reference notice itself. No additional near copies met
the protocol's conservative dHash+RGB rule.

English and Korean attribute sentences are in
`tmp/notice_extension_20260916/taiwan/evaluation/attribute_queries_bilingual.json`.
These are fixed dictionary renderings, not natural prose translations.

Original remarks are translated to English locally with two research models:

- [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M),
  revision `f8d333a098d19b4fd9a8b18f94170487ad3f821d`.
- [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), resolved
  revision stored in `translator_qwen.json`. Added after inspecting NLLB errors
  but before any Taiwan retrieval scores; see the translation amendment.

Neither translator receives photos or structured labels. Original remarks,
translations, model revisions, generation settings, and capped outputs are
retained in the ignored local evaluation directory. These are automatic
translations, not human-validated translations. Models stay in the research
environment; the application and deployed indexes are unchanged.

## Fixed comparisons

Frozen CLIP ViT-B/32 and DINOv3 ViT-B/16; original Korean-trained Linear, MLP and
Flow heads; text weight .20. No Taiwan training or test tuning. Relevance is
the exact normalized full color-token set plus Taiwan categorical size.
Report nDCG@10 and P@10, paired shelter-cluster bootstrap intervals, and shifted
text controls. Chinese, NLLB English and Qwen English use identical prose query
subsets. Full prose is still scored ONLY against appearance proxy labels.

See [protocol](TAIWAN_EVAL_PROTOCOL.md),
[translation amendment](TAIWAN_TRANSLATION_AMENDMENT.md), and
[results](TAIWAN_EVAL_RESULTS.md).

## Reproduce

Use the `dog-rag` environment from the repository root. The frozen snapshots and
translation models must already be available in the local cache.

```powershell
python -m experiments.dog_domain.taiwan_eval images
python -m experiments.dog_domain.taiwan_eval translate
python -m experiments.dog_domain.translate_taiwan_qwen
python -m experiments.dog_domain.taiwan_eval evaluate
python -m experiments.dog_domain.verify_taiwan_eval
```

`evaluate_base` optionally runs the first four conditions while Qwen translation
is pending. It writes separate `base_*` artifacts. The final evaluation verifies
their hashes and reuses those already computed conditions, then adds Qwen.

The verification script checks saved metric arithmetic for every query/method,
ensures the reference never appears in top ten, and independently recomputes
sampled image/native-CLIP rankings in float64. Confidence intervals quantify
sampling variability, not public-label or translation correctness.
