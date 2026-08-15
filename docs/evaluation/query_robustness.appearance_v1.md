# 외형 질의 개발셋 회귀·실패 사례 평가

- 생성 시각(UTC): `2026-08-14T13:01:21+00:00`
- 고정 기준일: `2026-07-26`
- 시스템: `appearance_natural_graph`
- 평가 역할: `development_regression_set`
- 평가 경로: 일반 retrieval headline과 구분된 실제 외형 모드 경로(외형 질의 정규화 후 natural hybrid+graph 검색, temperament 그래프 신호 제외)
- 판정 방식: 품질 합격선 없이 기술 통계만 공개하며, 비활성 공고 노출 0과 외형 모드의 생활·성격 문구 제거만 불변조건으로 검사
- **중요:** 최초 v1 실패를 확인한 뒤 정규화 코드를 보강했으므로 아래 수치는 독립 테스트가 아닌 개발셋 회귀 결과입니다. 일반화 주장을 하려면 별도로 동결한 미관측 한국어 holdout이 필요합니다.

## 요약

| 범위 | 질의 수 | P@5 | R@10 | nDCG@5 | MRR | Hit@5 | 비활성 노출 | Jaccard@10 | RBO@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 원 질의 | 12 | 0.9000 | 0.4182 | 0.9398 | 0.9444 | 1.0000 | 0.0000 | - | - |
| 의미 보존 변형 합계 | 48 | 0.9000 | 0.4182 | 0.9398 | 0.9444 | 1.0000 | 0.0000 | 1.0000 | 0.9997 |

## 변형 범주별 결과

| 범주 | 집계 포함 | 수 | P@5 | nDCG@5 | Hit@5 | Jaccard@5 | Jaccard@10 | RBO@10 | Top10 완전일치 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `synonym_paraphrase` | 예 | 12 | 0.9000 | 0.9398 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `word_order` | 예 | 12 | 0.9000 | 0.9398 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `conservative_typo` | 예 | 12 | 0.9000 | 0.9398 | 1.0000 | 1.0000 | 1.0000 | 0.9987 | 0.9167 |
| `lifestyle_personality_noise` | 예 | 12 | 0.9000 | 0.9398 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `negation_diagnostic` | 아니오 | 12 | 0.2667 | 0.2531 | 0.5833 | 0.0509 | 0.1690 | 0.1314 | 0.0000 |

`negation_diagnostic`은 원 질의와 의미가 다릅니다. 부정된 외형 조건은 양성 필터로 뒤집히지 않도록 검색 신호에서 제거되지만, 부정 의미 자체를 만족시키지는 않습니다. 원 qrels 대비 수치는 fail-safe 동작 관찰값일 뿐 부정 질의의 검색 정확도가 아닙니다.

## 가장 불안정한 의미 보존 변형

| 변형 ID | 범주 | Jaccard@10 | RBO@10 | ΔnDCG@5 | 관찰 사항 |
|---|---|---:|---:|---:|---|
| `black-medium::conservative_typo` | `conservative_typo` | 1.0000 | 0.9842 | 0.0000 | top10_order_changed |
| `adult-3-to-8kg::conservative_typo` | `conservative_typo` | 1.0000 | 1.0000 | 0.0000 | 없음 |
| `adult-3-to-8kg::lifestyle_personality_noise` | `lifestyle_personality_noise` | 1.0000 | 1.0000 | 0.0000 | 없음 |
| `adult-3-to-8kg::synonym_paraphrase` | `synonym_paraphrase` | 1.0000 | 1.0000 | 0.0000 | 없음 |
| `adult-3-to-8kg::word_order` | `word_order` | 1.0000 | 1.0000 | 0.0000 | 없음 |
| `black-medium::lifestyle_personality_noise` | `lifestyle_personality_noise` | 1.0000 | 1.0000 | 0.0000 | 없음 |
| `black-medium::synonym_paraphrase` | `synonym_paraphrase` | 1.0000 | 1.0000 | 0.0000 | 없음 |
| `black-medium::word_order` | `word_order` | 1.0000 | 1.0000 | 0.0000 | 없음 |
| `brown-small::conservative_typo` | `conservative_typo` | 1.0000 | 1.0000 | 0.0000 | 없음 |
| `brown-small::lifestyle_personality_noise` | `lifestyle_personality_noise` | 1.0000 | 1.0000 | 0.0000 | 없음 |
| `brown-small::synonym_paraphrase` | `synonym_paraphrase` | 1.0000 | 1.0000 | 0.0000 | 없음 |
| `brown-small::word_order` | `word_order` | 1.0000 | 1.0000 | 0.0000 | 없음 |

## 불변조건 검증

- 결과: **PASS**
- 비활성 공고 최대 노출률: `0.000000`
- 실패: `0`건

## 재현 근거

| 아티팩트 | SHA-256 |
|---|---|
| `index` | `2722aef8257b8748bd2cdd00d4224e7b4dee3d1d552a235439eb4c97c71aface` |
| `metas` | `723a4df01e2802ad5d42bd276eefbded097a0a6e72900784395c4c19626b6c00` |
| `base_queries` | `fb8713b31d1210ee72eeb1ceb2dca37cceb24777dc3f00142a9adec6d090c03d` |
| `variants` | `42b853f78dcfc21c254f0f8f9648511ea59c1610c6287c37ed3a5f1fc1423819` |
| `evaluation_module` | `724c27b7df3a4c43b9579cb0c01aa9a361801ed1331668e1c0ff202dbdac8f4a` |
| `evaluation_cli` | `ce205084a44b4dd1f2f86c12552ca339f64ca9ef30907a0490e21e758d7b0682` |
| `appearance_query_module` | `fa9b063d9d805ac450d9a531bc971d7144c18067b2e18bd722c2e2f5c1796d9f` |
| `hybrid_rag_module` | `4a2de17cbbb837ef88dd34c831b79f34942a385acd82d7ae54b24fd55eb03dfd` |
| `graph_rag_module` | `83039703d1f02f22769444b19b529cc863670a637a146b0249f57054ff642c81` |
| `retrieval_evaluation_module` | `86856f56ee6792d13e4f8a0d6ca895277417a3962712610778af39473dc48266` |
| `retrieval_evaluation_cli` | `2b344fdcba3498e2280ab5fc03081134cb6f9882bebb013c02d9f9f78ae65e8a` |

## 해석 한계

- Silver qrels come from public structured fields, not blinded human relevance labels.
- The same public metadata feeds qrels and parts of hybrid ranking, so scores are coupled.
- The observed v1 failures informed normalizer hardening, so this is a development regression set, not an independent test set.
- A separately frozen, unseen Korean holdout is required before claiming query robustness generalization.
- One Korean variant per category and query does not represent all user language.
- Synonym and word-order equivalence is manually designed, not independently annotated.
- Negation semantics remain unsupported; negated objective terms are removed fail-safe and excluded from every robustness aggregate.
- No adoption suitability, temperament, child-friendliness, or pet compatibility is measured.

이 평가는 사람의 관련성 판정이나 입양 적합성 검증이 아닙니다. v1 문장 자체는 유지했지만 그 실패가 현재 코드 개선에 사용되었으므로 독립 성능 검증으로 해석할 수 없습니다.
