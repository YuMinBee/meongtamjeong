# 외형 검색 독립 blind holdout 평가 v3

**Frozen-before-system-import one-shot 결과입니다. 측정 뒤 코드·질의 수정, 문구 교체, 확인 재실행을 하지 않았고 성능 합격선도 없습니다.**

- 기준일: `2026-07-26`
- 시스템: `appearance_natural_graph`
- 독립성: `author-independent-but-same-repo`
- frozen SHA-256: `8bb4f757fb41ea5a8466e79d6c32799dff6030056fa1c175f5d4089b6343601a`
- 실행 질의: base `12`, semantic `24`, negative `6`
- negative 6개는 모든 semantic aggregate와 inherited qrels에서 제외

## 의미보존 aggregate

| 지표 | base 12 평균 | holdout 24 평균 | 변화 |
|---|---:|---:|---:|
| `precision@5` | 0.900000 | 0.766667 | -0.133333 |
| `recall@5` | 0.303596 | 0.242089 | -0.061507 |
| `recall@10` | 0.418238 | 0.360012 | -0.058226 |
| `nDCG@5` | 0.939773 | 0.791549 | -0.148224 |
| `MRR` | 0.944444 | 0.829861 | -0.114583 |
| `hit@5` | 1.000000 | 1.000000 | 0.000000 |

- 평균 top-5 Jaccard: `0.576224`
- 평균 top-10 Jaccard: `0.577797`
- 평균 RBO@10: `0.646237`
- top-10 완전일치율: `0.125000`
- 실패·변화 사례: `21` / `24`

## 의미보존 질의별 결과

| ID | base | 문장 | 정규화 | P@5 | nDCG@5 | J@10 | 관찰 |
|---|---|---|---|---:|---:|---:|---|
| `v3-brown-small-a` | `brown-small` | 갈색 털인 작은 강아지 좀 보여 주세요 | 갈색 소형견 털인 좀 보여 주세요 | 1.000000 | 1.000000 | 0.250000 | top5_membership_changed, top10_membership_changed, top10_order_changed |
| `v3-brown-small-b` | `brown-small` | 덩치는 작은 편이고 털색은 갈색인 아이 찾아봐줘 | 갈색 소형견 덩치는 편이고 털색은 찾아봐줘 | 1.000000 | 1.000000 | 0.818182 | top10_membership_changed, top10_order_changed |
| `v3-white-tiny-a` | `white-tiny` | 하얀 털에 아주 조그마한 강아지 있나요? | 흰색 초소형견 있나요 | 1.000000 | 1.000000 | 0.818182 | top5_membership_changed, top10_membership_changed, top10_order_changed |
| `v3-white-tiny-b` | `white-tiny` | 초소형이면서 흰색인 아이로 보여줘요 | 흰색 초소형이면서 아이로 보여줘요 | 1.000000 | 1.000000 | 0.428571 | top5_membership_changed, top10_membership_changed, top10_order_changed |
| `v3-black-medium-a` | `black-medium` | 중간 체격에 까만 털을 가진 강아지 보여 줄래? | 검정색 중간 체격에 털을 가진 보여 줄래 강아지 | 0.600000 | 0.529635 | 0.818182 | top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5,MRR |
| `v3-black-medium-b` | `black-medium` | 검은색 중형 강아지로 찾아주세요 | 검정색 중형견 강아지로 | 0.800000 | 0.830420 | 0.818182 | top5_membership_changed, top10_membership_changed, top10_order_changed |
| `v3-cream-small-a` | `cream-small` | 작은 체구의 크림빛 강아지 있으면 보여줘 | 크림색 소형견 체구의 있으면 | 1.000000 | 1.000000 | 1.000000 | top10_order_changed |
| `v3-cream-small-b` | `cream-small` | 털색 크림, 크기는 소형인 아이 찾아 주실래요? | 소형견 털색 크림 크기는 찾아 주실래요 | 1.000000 | 1.000000 | 0.666667 | top5_membership_changed, top10_membership_changed, top10_order_changed |
| `v3-gyeonggi-small-adult-a` | `gyeonggi-small-adult` | 다 큰 작은 강아지 중 경기 지역에 있는 아이 보여줘 | 경기도 소형견 다 큰 지역에 | 0.400000 | 0.383566 | 0.000000 | top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5,MRR |
| `v3-gyeonggi-small-adult-b` | `gyeonggi-small-adult` | 경기도 보호소의 소형 성견을 찾고 있어요 | 경기도 소형견 성견 보호소의 찾고 있어요 | 1.000000 | 1.000000 | 0.000000 | top5_membership_changed, top10_membership_changed, top10_order_changed |
| `v3-seoul-small-puppy-a` | `seoul-small-puppy` | 서울에서 보호하는 어린 소형 강아지 보여 주세요 | 서울 소형견 어린 보호하는 보여 주세요 | 0.800000 | 1.000000 | 0.818182 | top10_membership_changed, top10_order_changed |
| `v3-seoul-small-puppy-b` | `seoul-small-puppy` | 작은 애기 강아지로, 지역은 서울이요 | 소형견 애기 강아지로 지역은 서울이요 | 0.200000 | 0.246302 | 0.176471 | top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5,MRR |
| `v3-jeju-white-puppy-a` | `jeju-white-puppy` | 제주에 있는 아기 강아지 중 하얀 아이 찾아줘요 | 제주 흰색 찾아줘요 강아지 | 0.400000 | 0.277273 | 0.538462 | top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5,MRR |
| `v3-jeju-white-puppy-b` | `jeju-white-puppy` | 흰 털의 어린 강아지, 제주 지역으로 보여 주세요 | 제주 흰색 어린 털의 지역으로 보여 주세요 강아지 | 1.000000 | 1.000000 | 0.818182 | top5_membership_changed, top10_membership_changed, top10_order_changed |
| `v3-jeonbuk-brown-large-a` | `jeonbuk-brown-large` | 전라북도 쪽 갈색 대형 강아지를 찾습니다 | 전북 갈색 대형견 쪽 찾습니다 | 0.600000 | 1.000000 | 0.250000 | top5_membership_changed, top10_membership_changed, top10_order_changed |
| `v3-jeonbuk-brown-large-b` | `jeonbuk-brown-large` | 덩치 큰 갈색 아이로 전북에 있는 강아지 보여줘 | 전북 갈색 덩치 큰 아이로 강아지 | 0.600000 | 0.967468 | 0.250000 | top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:nDCG@5 |
| `v3-gyeongnam-black-small-a` | `gyeongnam-black-small` | 경상남도에서 보호 중인 까만 소형견 있나요? | 경남 검정색 소형견 있나요 | 1.000000 | 1.000000 | 1.000000 | 없음 |
| `v3-gyeongnam-black-small-b` | `gyeongnam-black-small` | 검은 털에 작은 강아지, 경남 지역으로 찾아 줘요 | 경남 검정색 소형견 털에 지역으로 찾아 줘요 | 1.000000 | 1.000000 | 1.000000 | top5_membership_changed, top10_order_changed |
| `v3-gyeonggi-tan-medium-a` | `gyeonggi-tan-medium` | 경기 지역의 중간 크기 황갈색 강아지 보여주세요 | 경기도 황갈색 중형견 | 1.000000 | 1.000000 | 1.000000 | 없음 |
| `v3-gyeonggi-tan-medium-b` | `gyeonggi-tan-medium` | 털은 누르스름한 갈색이고 중형인 아이, 경기도로 찾아줘 | 갈색 중형견 털은 누르스름한 경기도로 | 0.200000 | 0.339160 | 0.052632 | top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5 |
| `v3-adult-3-to-8kg-a` | `adult-3-to-8kg` | 성견 중 몸무게 3~8 kg인 아이 찾아줘 | 성견 3kg 이상 8kg 이하 | 0.600000 | 0.446854 | 1.000000 | 없음 |
| `v3-adult-3-to-8kg-b` | `adult-3-to-8kg` | 다 큰 강아지로 3키로 이상 8키로 이하만 보여 주세요 | 성견로 3kg 이상 8kg 이하 보여 주세요 | 0.600000 | 0.446854 | 0.250000 | top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:recall@10 |
| `v3-senior-under-8kg-a` | `senior-under-8kg` | 나이 든 강아지 중 8kg 안 넘는 작은 아이 보여줘요 | 소형견 노령견 8kg 안 넘는 보여줘요 | 1.000000 | 1.000000 | 0.666667 | top5_membership_changed, top10_membership_changed, top10_order_changed |
| `v3-senior-under-8kg-b` | `senior-under-8kg` | 몸무게 여덟 키로 이하인 노령견을 찾고 있어요 | 노령견 여덟 키로 이하 찾고 있어요 | 0.600000 | 0.529635 | 0.428571 | top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5,MRR |

## 모든 실패·변화 사례

- `v3-gyeonggi-small-adult-a` — 다 큰 작은 강아지 중 경기 지역에 있는 아이 보여줘: top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5,MRR
- `v3-gyeonggi-small-adult-b` — 경기도 보호소의 소형 성견을 찾고 있어요: top5_membership_changed, top10_membership_changed, top10_order_changed
- `v3-gyeonggi-tan-medium-b` — 털은 누르스름한 갈색이고 중형인 아이, 경기도로 찾아줘: top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5
- `v3-seoul-small-puppy-b` — 작은 애기 강아지로, 지역은 서울이요: top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5,MRR
- `v3-brown-small-a` — 갈색 털인 작은 강아지 좀 보여 주세요: top5_membership_changed, top10_membership_changed, top10_order_changed
- `v3-jeonbuk-brown-large-b` — 덩치 큰 갈색 아이로 전북에 있는 강아지 보여줘: top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:nDCG@5
- `v3-adult-3-to-8kg-b` — 다 큰 강아지로 3키로 이상 8키로 이하만 보여 주세요: top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:recall@10
- `v3-jeonbuk-brown-large-a` — 전라북도 쪽 갈색 대형 강아지를 찾습니다: top5_membership_changed, top10_membership_changed, top10_order_changed
- `v3-senior-under-8kg-b` — 몸무게 여덟 키로 이하인 노령견을 찾고 있어요: top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5,MRR
- `v3-white-tiny-b` — 초소형이면서 흰색인 아이로 보여줘요: top5_membership_changed, top10_membership_changed, top10_order_changed
- `v3-jeju-white-puppy-a` — 제주에 있는 아기 강아지 중 하얀 아이 찾아줘요: top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5,MRR
- `v3-senior-under-8kg-a` — 나이 든 강아지 중 8kg 안 넘는 작은 아이 보여줘요: top5_membership_changed, top10_membership_changed, top10_order_changed
- `v3-cream-small-b` — 털색 크림, 크기는 소형인 아이 찾아 주실래요?: top5_membership_changed, top10_membership_changed, top10_order_changed
- `v3-black-medium-a` — 중간 체격에 까만 털을 가진 강아지 보여 줄래?: top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5,MRR
- `v3-jeju-white-puppy-b` — 흰 털의 어린 강아지, 제주 지역으로 보여 주세요: top5_membership_changed, top10_membership_changed, top10_order_changed
- `v3-white-tiny-a` — 하얀 털에 아주 조그마한 강아지 있나요?: top5_membership_changed, top10_membership_changed, top10_order_changed
- `v3-brown-small-b` — 덩치는 작은 편이고 털색은 갈색인 아이 찾아봐줘: top10_membership_changed, top10_order_changed
- `v3-black-medium-b` — 검은색 중형 강아지로 찾아주세요: top5_membership_changed, top10_membership_changed, top10_order_changed
- `v3-seoul-small-puppy-a` — 서울에서 보호하는 어린 소형 강아지 보여 주세요: top10_membership_changed, top10_order_changed
- `v3-gyeongnam-black-small-b` — 검은 털에 작은 강아지, 경남 지역으로 찾아 줘요: top5_membership_changed, top10_order_changed
- `v3-cream-small-a` — 작은 체구의 크림빛 강아지 있으면 보여줘: top10_order_changed

## 부정 진단 — aggregate 제외

positive silver qrels를 상속하지 않으며, 사전 정의된 오적용 진단만 기록합니다.
- 진단 PASS `6` / `6`, 실패 `0`

| ID | 문장 | 정규화 | 진단 | 결과 | 실패 |
|---|---|---|---|---:|---|
| `v3-neg-excluded-color` | 갈색은 빼고 작은 강아지 찾아줘 | 소형견 | `negated_color_must_not_become_positive` | PASS | 없음 |
| `v3-neg-contrast-region` | 서울 말고 경기의 어린 강아지 보여줘 | 경기도 어린 강아지 | `negated_region_must_not_survive_contrast` | PASS | 없음 |
| `v3-neg-opposite-weight` | 8kg 이상인 노령견을 찾고 있어요 | 노령견 8kg 이상 찾고 있어요 | `lower_bound_must_not_become_upper_bound` | PASS | 없음 |
| `v3-neg-no-appearance` | 강아지 좀 보여줘 | 좀 강아지 | `generic_query_must_not_invent_constraints` | PASS | 없음 |
| `v3-neg-unsupported-traits` | 성격이 온순하고 사람을 잘 따르는 아이 찾아줘 | 따르는 | `unsupported_traits_must_not_invent_constraints` | PASS | 없음 |
| `v3-neg-other-species` | 고양이를 찾고 있어요 | 찾고 있어요 | `other_species_must_not_invent_dog_appearance_constraints` | PASS | 없음 |

## 무결성·비활성 노출

- 무결성: **PASS**
- 범위: `integrity_invariants_only_not_performance_acceptance`
- 무결성 실패: `0`건
- base 최대: `0.000000`
- semantic 최대: `0.000000`
- negative 최대: `0.000000`
- 전체 최대: `0.000000`
- 비활성 노출 질의 수: `0`

## 시스템 코드 SHA-256

- bundle: `ab70f8d43014afd31c1d8cd0de4095873470b983cb0458a5309ef06f2c13d26d`

| 코드 아티팩트 | SHA-256 |
|---|---|
| `appearance_query_module` | `fa9b063d9d805ac450d9a531bc971d7144c18067b2e18bd722c2e2f5c1796d9f` |
| `hybrid_rag_module` | `2e577c9a09ceaba058c37de5c6198f48dcbf2c2f94c86774e3156048f45d710a` |
| `graph_rag_module` | `83039703d1f02f22769444b19b529cc863670a637a146b0249f57054ff642c81` |
| `retrieval_evaluation_module` | `86856f56ee6792d13e4f8a0d6ca895277417a3962712610778af39473dc48266` |
| `retrieval_evaluation_cli` | `2b344fdcba3498e2280ab5fc03081134cb6f9882bebb013c02d9f9f78ae65e8a` |

## 전체 아티팩트 SHA-256

| 아티팩트 | SHA-256 |
|---|---|
| `index` | `2722aef8257b8748bd2cdd00d4224e7b4dee3d1d552a235439eb4c97c71aface` |
| `metas` | `723a4df01e2802ad5d42bd276eefbded097a0a6e72900784395c4c19626b6c00` |
| `base_queries` | `fb8713b31d1210ee72eeb1ceb2dca37cceb24777dc3f00142a9adec6d090c03d` |
| `holdout` | `8bb4f757fb41ea5a8466e79d6c32799dff6030056fa1c175f5d4089b6343601a` |
| `holdout_sha256` | `2f1abad84967b9bf1b0ca2b0c4766ae079a5e50d7f07047dda8149cf4f0d85ac` |
| `evaluation_module` | `aafaa48467e013be32886960199949d4eecf6a37389a6a3e79e7d53005a36c5f` |
| `evaluation_cli` | `7a74b8bc956c9fc812b27cbc2a289a2eabf3155b55eaedc0071c4ae96f8d89df` |
| `appearance_query_module` | `fa9b063d9d805ac450d9a531bc971d7144c18067b2e18bd722c2e2f5c1796d9f` |
| `hybrid_rag_module` | `2e577c9a09ceaba058c37de5c6198f48dcbf2c2f94c86774e3156048f45d710a` |
| `graph_rag_module` | `83039703d1f02f22769444b19b529cc863670a637a146b0249f57054ff642c81` |
| `retrieval_evaluation_module` | `86856f56ee6792d13e4f8a0d6ca895277417a3962712610778af39473dc48266` |
| `retrieval_evaluation_cli` | `2b344fdcba3498e2280ab5fc03081134cb6f9882bebb013c02d9f9f78ae65e8a` |

## 한계

- author-independent-but-same-repo: the holdout wording was frozen without inspecting the appearance-query implementation, its tests, prior variant/holdout wording, or prior query-evaluation reports; shared repository context and inherited base intents still limit independence
- The 24 semantic cases are a small manually authored Korean sample with two variants per base intent; no confidence interval or population-level generalization claim is valid
- Silver qrels derive from public notice color, weight, age, region, and status fields, not independent human relevance judgments
- Negative requests do not inherit positive qrels and are diagnostic only; their six outcomes are excluded from every semantic aggregate
- This is one frozen one-shot execution with no post-measurement code tuning, query replacement, or confirmatory search rerun
- Integrity PASS has no performance floor and does not establish adoption suitability, human relevance, or external replication

이 평가는 사람의 관련성 판정, 입양 적합성 평가, 외부 저자에 의한 완전 독립 검증이 아닙니다.
