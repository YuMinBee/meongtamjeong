# Translation diagnostic amendment — before any Taiwan retrieval scores

A deterministic sample of 12 SHA256-sorted original remarks revealed material
NLLB errors, including sexual meaning added to a warning about aggressive
behavior, sterilization mistranslated as abortion, and a collar as a circle.
The sample selection and raw outputs are saved in `translation_spotcheck.json`.
Six generated NLLB segments also reached the output cap.

Keep the NLLB translation condition and add a second, explicitly exploratory
condition using `Qwen/Qwen2.5-3B-Instruct`, pinned revision. This amendment is
fixed before retrieval scores, based solely on translation inspection.
Translate the same unique full original remarks to English using only the text
and a fixed dog-adoption translation instruction. Never supply images, public
attribute labels, model scores, or relevance judgments. No selective correction
or translator selection by retrieval performance.

Use BF16 local inference, greedy decoding (do_sample=False, num_beams=1),
max_new_tokens=768, batch size 8, no source truncation. Save the exact system
prompt, model revision, source hash, and capped output count. Retain NLLB outputs
unchanged. Spot-check the same 12 texts, but do not claim human validation.

All other benchmark rules and template primary contrasts stay as in
TAIWAN_EVAL_PROTOCOL.md. The added English prose condition uses precisely the
same query subset and gallery as Chinese and NLLB prose. Compare its metrics
against Chinese, NLLB, DINO image-only, and fixed fusion baselines. Prose scores
still measure only appearance proxy relevance, not translation accuracy or
full-description semantic correctness.
