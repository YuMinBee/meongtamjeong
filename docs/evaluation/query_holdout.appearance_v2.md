# 외형 검색 독립 holdout 평가 v2

**이 결과는 frozen-before-run one-shot 측정입니다. 측정 뒤 코드·질의 튜닝이나 확인용 재실행을 하지 않았으며, 성능 합격선을 두지 않았습니다.**

- 기준일: `2026-07-26`
- 시스템: `appearance_natural_graph`
- 독립성 범위: `author-independent-but-same-repo`
- 고정 holdout SHA-256: `6b74bf0082e4911b8db6a5a36205b43d316f77b98162d81c566fca3fffaa65e6`
- 의미보존 질의: `12`개
- 부정 진단: `3`개 (의미보존 aggregate에서 제외)
- 검증 PASS는 파일·집계 불변조건만 뜻하며 성능 합격을 뜻하지 않음

## 의미보존 aggregate

| 지표 | base 평균 | holdout 평균 | 변화 |
|---|---:|---:|---:|
| `precision@5` | 0.900000 | 0.483333 | -0.416667 |
| `recall@5` | 0.303596 | 0.231136 | -0.072460 |
| `recall@10` | 0.418238 | 0.303462 | -0.114776 |
| `nDCG@5` | 0.939773 | 0.514634 | -0.425139 |
| `MRR` | 0.944444 | 0.545000 | -0.399444 |
| `hit@5` | 1.000000 | 0.666667 | -0.333333 |

- 평균 top-5 Jaccard: `0.224537`
- 평균 top-10 Jaccard: `0.219907`
- 평균 RBO@10: `0.276126`
- top-10 완전일치율: `0.000000`

## 의미보존 질의별 결과

| ID | holdout 문장 | 정규화 문장 | P@5 | nDCG@5 | top10 Jaccard | 관찰 |
|---|---|---|---:|---:|---:|---|
| `holdout-brown-small-colloquial` | 쪼꼬만 밤색 강쥐 있나여?? | 쪼꼬만 밤색 강쥐 있나여 | 0.200000 | 0.213986 | 0.000000 | top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5,MRR |
| `holdout-white-tiny-order` | 하얀 털에 아주 조그마한 개로요 | 흰색 털에 아주 조그마한 개로요 | 0.000000 | 0.000000 | 0.000000 | top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5,MRR,hit@5 |
| `holdout-black-medium-slang` | 중간 덩치의 검은 털 댕댕이 보여주실래요 | 검정색 중간 덩치의 털 댕댕이 보여주실래요 | 1.000000 | 1.000000 | 0.111111 | top5_membership_changed, top10_membership_changed, top10_order_changed |
| `holdout-cream-small-spacing` | 소형에 크림 빛 털인 강아지 찾아 주세요!!! | 소형견 크림 빛 털인 찾아 주세요 | 1.000000 | 1.000000 | 0.666667 | top5_membership_changed, top10_membership_changed, top10_order_changed |
| `holdout-gyeonggi-small-adult-fragment` | 다 큰 작은 강아지, 경기 쪽으로요 | 경기도 소형견 다 큰 쪽으로요 | 0.600000 | 0.514771 | 0.000000 | top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5,MRR |
| `holdout-seoul-small-puppy-colloquial` | 새끼 강쥐 작은 애로 서울에서 찾아줘용 | 서울 소형견 어린 강쥐 애로 찾아줘용 | 0.800000 | 1.000000 | 1.000000 | top10_order_changed |
| `holdout-jeju-white-puppy-compound` | 흰 털 아기개 제주 보호소꺼 있나요? | 제주 흰색 털 보호소꺼 있나요 | 0.600000 | 0.446854 | 0.250000 | top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5,MRR |
| `holdout-jeonbuk-brown-large-expanded` | 큰 밤색 개, 전라북도 쪽 | 전북 큰 밤색 쪽 강아지 | 0.600000 | 1.000000 | 0.250000 | top5_membership_changed, top10_membership_changed, top10_order_changed |
| `holdout-gyeongnam-black-small-no-space` | 까만털 작은개요.. 경상남도에 있는 | 경남 까만털 작은개요 | 1.000000 | 1.000000 | 0.250000 | top5_membership_changed, top10_membership_changed, top10_order_changed |
| `holdout-gyeonggi-tan-medium-no-space` | 경기쪽 중간덩치 누런갈색 강아지 | 경기도 중간덩치 누런갈색 강아지 | 0.000000 | 0.000000 | 0.111111 | top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5,MRR,hit@5 |
| `holdout-adult-3-to-8kg-unit` | 3키로부터 8키로까지, 다 큰 강아지 | 대형견 3키로부터 8키로까지 다 | 0.000000 | 0.000000 | 0.000000 | top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5,MRR,hit@5 |
| `holdout-senior-under-8kg-no-space` | 8키로 이하 나이많은 쪼그만 개 | 8키로 이하 나이많은 쪼그만 강아지 | 0.000000 | 0.000000 | 0.000000 | top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5,MRR,hit@5 |

## 부정 질의 진단 — aggregate 제외

부정 질의는 원 qrels와 의미가 다르므로 precision/recall을 계산하지 않았습니다. 아래는 fail-safe 신호와 원 긍정 질의 대비 순위만 진단합니다.

| ID | 문장 | 정규화 문장 | fail-safe 신호 | top10 Jaccard | 관찰 |
|---|---|---|---:|---:|---|
| `negative-exclude-brown-small` | 갈색 소형견은 빼고 찾아줘 | 갈색 | 있음 | 0.250000 | 없음 |
| `negative-exclude-seoul` | 서울 말고 어린 소형견을 보여줘 | 소형견 어린 | 있음 | 0.666667 | 없음 |
| `negative-exclude-senior-under-8kg` | 8kg 이하 노령견은 제외해줘 | 8kg 이하 줘 | 있음 | 0.000000 | 없음 |

## 실패·변화 사례

- `holdout-senior-under-8kg-no-space` — 8키로 이하 나이많은 쪼그만 개: top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5,MRR,hit@5
- `holdout-white-tiny-order` — 하얀 털에 아주 조그마한 개로요: top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5,MRR,hit@5
- `holdout-brown-small-colloquial` — 쪼꼬만 밤색 강쥐 있나여??: top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5,MRR
- `holdout-gyeonggi-small-adult-fragment` — 다 큰 작은 강아지, 경기 쪽으로요: top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5,MRR
- `holdout-adult-3-to-8kg-unit` — 3키로부터 8키로까지, 다 큰 강아지: top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5,MRR,hit@5
- `holdout-gyeonggi-tan-medium-no-space` — 경기쪽 중간덩치 누런갈색 강아지: top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5,MRR,hit@5
- `holdout-black-medium-slang` — 중간 덩치의 검은 털 댕댕이 보여주실래요: top5_membership_changed, top10_membership_changed, top10_order_changed
- `holdout-gyeongnam-black-small-no-space` — 까만털 작은개요.. 경상남도에 있는: top5_membership_changed, top10_membership_changed, top10_order_changed
- `holdout-jeju-white-puppy-compound` — 흰 털 아기개 제주 보호소꺼 있나요?: top5_membership_changed, top10_membership_changed, top10_order_changed, silver_metric_decreased:precision@5,recall@5,recall@10,nDCG@5,MRR
- `holdout-jeonbuk-brown-large-expanded` — 큰 밤색 개, 전라북도 쪽: top5_membership_changed, top10_membership_changed, top10_order_changed
- `holdout-cream-small-spacing` — 소형에 크림 빛 털인 강아지 찾아 주세요!!!: top5_membership_changed, top10_membership_changed, top10_order_changed
- `holdout-seoul-small-puppy-colloquial` — 새끼 강쥐 작은 애로 서울에서 찾아줘용: top10_order_changed

## 무결성 검증

- 결과: **PASS**
- 범위: `integrity_invariants_only_not_performance_acceptance`
- 실패: `0`건
- 비활성 공고 최대 노출률: `0.000000`

## 재현 근거

| 아티팩트 | SHA-256 |
|---|---|
| `index` | `2722aef8257b8748bd2cdd00d4224e7b4dee3d1d552a235439eb4c97c71aface` |
| `metas` | `723a4df01e2802ad5d42bd276eefbded097a0a6e72900784395c4c19626b6c00` |
| `base_queries` | `fb8713b31d1210ee72eeb1ceb2dca37cceb24777dc3f00142a9adec6d090c03d` |
| `holdout` | `6b74bf0082e4911b8db6a5a36205b43d316f77b98162d81c566fca3fffaa65e6` |
| `holdout_sha256` | `55c971872596828de0017d374b4ebf107677d84889a5159153620c77fb0e5f74` |
| `evaluation_module` | `f8ed809a84b2165a519bc7fc5e97504d353e60dadc1e21f58cbec81b9291fe61` |
| `evaluation_cli` | `1be260cfebe4d6a3da3d66e2684fde944ace566b5ad77a30a00b031af8f1faf4` |
| `hybrid_rag_module` | `2e577c9a09ceaba058c37de5c6198f48dcbf2c2f94c86774e3156048f45d710a` |
| `graph_rag_module` | `83039703d1f02f22769444b19b529cc863670a637a146b0249f57054ff642c81` |
| `retrieval_evaluation_module` | `86856f56ee6792d13e4f8a0d6ca895277417a3962712610778af39473dc48266` |
| `retrieval_evaluation_cli` | `2b344fdcba3498e2280ab5fc03081134cb6f9882bebb013c02d9f9f78ae65e8a` |

## 한계

- author-independent-but-same-repo: the holdout author did not inspect the excluded v1 variants or appearance-query implementation, but shared repository context and base qrels can still align assumptions
- The 12 semantic cases are a small manually authored sample with one case per base intent; no confidence interval or population claim is valid
- Silver qrels come from public notice color, weight, age, region, and status fields rather than independent human relevance judgments
- Negated requests change semantics, so they are diagnostics only and are not scored against inherited positive qrels
- This is one frozen one-shot run; no code or query tuning and no confirmatory rerun followed measurement
- The blinded appearance-query module is invoked at runtime but its content hash is intentionally omitted from author-visible provenance

이 평가는 사람의 관련성 판정, 입양 적합성 평가, 외부 저자에 의한 완전 독립 검증이 아닙니다.
