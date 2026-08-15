"""Pure helpers for reproducible retrieval evaluation.

The silver relevance labels in this module are derived only from structured
public-notice fields.  Retrieval result text, descriptions, embeddings, and
ranked output are deliberately excluded from the labeling path.
"""

from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from app.graph_rag import infer_age_hint, infer_region, infer_size_from_weight
from app.notice_status import classify_notice


REPORT_SCHEMA_VERSION = "retrieval-evaluation.v2"
SILVER_LABEL_VERSION = "public-notice-appearance.v1"
HEADLINE_SYSTEM_ID = "hybrid_natural_graph"
BASELINE_SYSTEM_ID = "clip_active"
ORACLE_SYSTEM_ID = "hybrid_structured_oracle_graph"
DIAGNOSTIC_SYSTEM_ID = "clip_raw_unfiltered"
METRIC_KEYS = (
    "precision@5",
    "recall@5",
    "recall@10",
    "nDCG@5",
    "MRR",
    "hit@5",
    "inactive_exposure",
)
SUPPORTED_CRITERIA = {
    "colors_any",
    "sizes_any",
    "age_hints_any",
    "regions_any",
    "weight_kg",
    "active_only",
}

_UNKNOWN_VALUES = {
    "",
    "-",
    "--",
    "unknown",
    "none",
    "null",
    "n/a",
    "미상",
    "알 수 없음",
    "정보 없음",
}
_WEIGHT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)")
_COLOR_SPLIT_RE = re.compile(r"[\s,&/+·]+")
_PUBLIC_FIELD_ALIASES = {
    "id": ("desertionNo", "desertion_no"),
    "color": ("color", "colorCd"),
    "age": ("age",),
    "weight": ("weight",),
    "region": (
        "region",
        "org_name",
        "orgNm",
        "care_addr",
        "careAddr",
        "happen_place",
        "happenPlace",
        "care_name",
        "careNm",
    ),
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_known(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        return any(is_known(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(is_known(item) for item in value)
    return clean_text(value).lower() not in _UNKNOWN_VALUES


def parse_reference_date(value: Any) -> datetime:
    """Parse an explicit evaluation date; never silently use today's date."""

    text = clean_text(value)
    if not text:
        raise ValueError("reference_date is required for reproducible evaluation")
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"invalid reference_date: {text!r}")


def notice_id(meta: Mapping[str, Any], fallback: str = "") -> str:
    for key in _PUBLIC_FIELD_ALIASES["id"]:
        value = clean_text(meta.get(key))
        if value:
            return value
    return fallback


def merge_notice_metas(
    metas: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Collapse text/image/crop vectors into one public notice per ID."""

    merged: dict[str, dict[str, Any]] = {}
    for index, raw_meta in enumerate(metas):
        if not isinstance(raw_meta, Mapping):
            continue
        dog_id = notice_id(raw_meta, fallback=f"idx-{index}")
        current = merged.setdefault(dog_id, {})
        prefer = clean_text(raw_meta.get("type")) == "image"
        for key, value in raw_meta.items():
            if not is_known(value):
                continue
            if key not in current or not is_known(current.get(key)) or prefer:
                current[key] = value
        current.setdefault("desertionNo", dog_id)
    return merged


def parse_weight_kg(value: Any) -> float | None:
    text = clean_text(value).replace(",", "")
    match = _WEIGHT_RE.search(text)
    if not match:
        return None
    weight = float(match.group(1))
    if not math.isfinite(weight) or weight < 0:
        return None
    return weight


def normalize_public_colors(value: Any) -> tuple[str, ...]:
    """Normalize only the notice's reported color field into coarse labels."""

    text = clean_text(value).lower()
    if not text or text in _UNKNOWN_VALUES:
        return ()

    labels: set[str] = set()
    tokens = [token for token in _COLOR_SPLIT_RE.split(text) if token]
    for token in tokens:
        if any(term in token for term in ("흰", "백색", "하얀", "화이트", "white")):
            labels.add("white")
        if any(
            term in token for term in ("검정", "검은", "까만", "흑색", "블랙", "black")
        ):
            labels.add("black")
        if any(term in token for term in ("크림", "아이보리", "cream", "ivory")):
            labels.add("cream")
        if any(term in token for term in ("회색", "그레이", "gray", "grey")):
            labels.add("gray")
        if any(term in token for term in ("황갈", "탄색", "tan")):
            labels.add("tan")
        if (
            any(
                term in token
                for term in (
                    "갈색",
                    "연갈",
                    "엷은갈",
                    "금갈",
                    "브라운",
                    "초콜릿",
                    "쵸콜릿",
                    "brown",
                )
            )
            and "황갈" not in token
        ):
            labels.add("brown")
        if any(term in token for term in ("노랑", "황색", "레몬", "yellow")):
            labels.add("yellow")
        if any(term in token for term in ("금색", "gold")):
            labels.add("gold")
        if any(term in token for term in ("반점", "얼룩", "spotted")):
            labels.add("spotted")
    return tuple(sorted(labels))


def normalize_region(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    return infer_region({"region": text})


def extract_public_attributes(
    meta: Mapping[str, Any],
    reference_date: datetime,
) -> dict[str, Any]:
    """Extract label inputs from public fields, without descriptions or VLM data."""

    color = next(
        (
            meta.get(key)
            for key in _PUBLIC_FIELD_ALIASES["color"]
            if is_known(meta.get(key))
        ),
        None,
    )
    age = next(
        (
            meta.get(key)
            for key in _PUBLIC_FIELD_ALIASES["age"]
            if is_known(meta.get(key))
        ),
        None,
    )
    weight = next(
        (
            meta.get(key)
            for key in _PUBLIC_FIELD_ALIASES["weight"]
            if is_known(meta.get(key))
        ),
        None,
    )
    weight_kg = parse_weight_kg(weight)
    size_hint = infer_size_from_weight(weight) if weight_kg is not None else ""
    region = infer_region(dict(meta))
    return {
        "dog_id": notice_id(meta),
        "colors": normalize_public_colors(color),
        "weight_kg": weight_kg,
        "size_hint": size_hint,
        "age_hint": infer_age_hint(age, reference_year=reference_date.year),
        "region": region,
        "notice_status": classify_notice(
            dict(meta),
            reference_date=reference_date,
        ),
    }


def _normalized_values(value: Any) -> set[str]:
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, Iterable):
        values = value
    else:
        values = ()
    return {clean_text(item).lower() for item in values if clean_text(item)}


def validate_query_specs(queries: Sequence[Mapping[str, Any]]) -> None:
    if not queries:
        raise ValueError("at least one evaluation query is required")
    seen: set[str] = set()
    for position, query in enumerate(queries):
        query_id = clean_text(query.get("id"))
        raw_query = clean_text(query.get("query"))
        criteria = query.get("criteria")
        if not query_id:
            raise ValueError(f"query at position {position} has no id")
        if query_id in seen:
            raise ValueError(f"duplicate query id: {query_id}")
        if not raw_query:
            raise ValueError(f"query {query_id!r} has no query text")
        if not isinstance(criteria, Mapping):
            raise ValueError(f"query {query_id!r} has no criteria object")
        unsupported = set(criteria) - SUPPORTED_CRITERIA
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"query {query_id!r} has unsupported criteria: {names}")
        label_dimensions = set(criteria) - {"active_only"}
        if not label_dimensions:
            raise ValueError(
                f"query {query_id!r} needs an appearance/location criterion"
            )
        seen.add(query_id)


def matches_silver_criteria(
    attributes: Mapping[str, Any],
    criteria: Mapping[str, Any],
) -> bool:
    if criteria.get("active_only", True):
        if attributes.get("notice_status") != "active":
            return False

    wanted_colors = _normalized_values(criteria.get("colors_any"))
    actual_colors = _normalized_values(attributes.get("colors"))
    if wanted_colors and not wanted_colors.intersection(actual_colors):
        return False

    wanted_sizes = _normalized_values(criteria.get("sizes_any"))
    if (
        wanted_sizes
        and clean_text(attributes.get("size_hint")).lower() not in wanted_sizes
    ):
        return False

    wanted_ages = _normalized_values(criteria.get("age_hints_any"))
    if (
        wanted_ages
        and clean_text(attributes.get("age_hint")).lower() not in wanted_ages
    ):
        return False

    wanted_regions = {
        normalize_region(value) or clean_text(value)
        for value in _normalized_values(criteria.get("regions_any"))
    }
    if wanted_regions and clean_text(attributes.get("region")) not in wanted_regions:
        return False

    weight_range = criteria.get("weight_kg")
    if weight_range is not None:
        if not isinstance(weight_range, Mapping):
            return False
        weight = attributes.get("weight_kg")
        if not isinstance(weight, (int, float)):
            return False
        minimum = weight_range.get("min")
        maximum = weight_range.get("max")
        if minimum is not None and weight < float(minimum):
            return False
        if maximum is not None and weight > float(maximum):
            return False
    return True


def build_silver_qrels(
    metas: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    reference_date: datetime,
) -> dict[str, set[str]]:
    """Build binary qrels before retrieval from public notice fields alone."""

    validate_query_specs(queries)
    notices = merge_notice_metas(metas)
    attributes_by_id = {
        dog_id: extract_public_attributes(meta, reference_date)
        for dog_id, meta in notices.items()
    }
    qrels: dict[str, set[str]] = {}
    for query in queries:
        query_id = clean_text(query.get("id"))
        criteria = query["criteria"]
        qrels[query_id] = {
            dog_id
            for dog_id, attributes in attributes_by_id.items()
            if matches_silver_criteria(attributes, criteria)
        }
    return qrels


def validate_nonempty_qrels(qrels: Mapping[str, set[str]]) -> None:
    empty = sorted(query_id for query_id, ids in qrels.items() if not ids)
    if empty:
        raise ValueError(
            "silver qrels contain no relevant documents for: " + ", ".join(empty)
        )


def deduplicate_ids(ranked_ids: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in ranked_ids:
        dog_id = clean_text(value)
        if dog_id and dog_id not in seen:
            output.append(dog_id)
            seen.add(dog_id)
    return output


def _dcg(relevance: Sequence[int]) -> float:
    return sum(
        value / math.log2(rank + 1) for rank, value in enumerate(relevance, start=1)
    )


def query_metrics(
    ranked_ids: Sequence[Any],
    relevant_ids: Iterable[Any],
    metas_by_id: Mapping[str, Mapping[str, Any]],
    reference_date: datetime,
) -> dict[str, float]:
    ranked = deduplicate_ids(ranked_ids)
    relevant = {clean_text(value) for value in relevant_ids if clean_text(value)}

    top5 = ranked[:5]
    top10 = ranked[:10]
    hits5 = sum(dog_id in relevant for dog_id in top5)
    hits10 = sum(dog_id in relevant for dog_id in top10)
    relevant_count = len(relevant)
    ideal = [1] * min(relevant_count, 5)
    actual = [int(dog_id in relevant) for dog_id in top5]
    ideal_dcg = _dcg(ideal)

    first_relevant_rank = next(
        (rank for rank, dog_id in enumerate(ranked, start=1) if dog_id in relevant),
        None,
    )
    inactive = 0
    evaluated = 0
    for dog_id in top10:
        meta = metas_by_id.get(dog_id)
        if not isinstance(meta, Mapping):
            continue
        evaluated += 1
        status = classify_notice(dict(meta), reference_date=reference_date)
        if status in {"closed", "expired"}:
            inactive += 1

    return {
        "precision@5": round(hits5 / 5.0, 6),
        "recall@5": round(hits5 / relevant_count, 6) if relevant_count else 0.0,
        "recall@10": round(hits10 / relevant_count, 6) if relevant_count else 0.0,
        "nDCG@5": round(_dcg(actual) / ideal_dcg, 6) if ideal_dcg else 0.0,
        "MRR": round(1.0 / first_relevant_rank, 6) if first_relevant_rank else 0.0,
        "hit@5": float(hits5 > 0),
        "inactive_exposure": round(inactive / evaluated, 6) if evaluated else 0.0,
    }


def build_query_result(
    query: Mapping[str, Any],
    ranked_ids: Sequence[Any],
    relevant_ids: Iterable[Any],
    metas_by_id: Mapping[str, Mapping[str, Any]],
    reference_date: datetime,
    latency_ms: float,
    top_ids_limit: int = 10,
) -> dict[str, Any]:
    relevant = {clean_text(value) for value in relevant_ids if clean_text(value)}
    ranked = deduplicate_ids(ranked_ids)
    return {
        "id": clean_text(query.get("id")),
        "query": clean_text(query.get("query")),
        "criteria": dict(query.get("criteria") or {}),
        "relevant_count": len(relevant),
        "relevant_ids": sorted(relevant),
        "top_ids": ranked[: max(1, int(top_ids_limit))],
        "latency_ms": round(float(latency_ms), 3),
        "metrics": query_metrics(
            ranked,
            relevant,
            metas_by_id,
            reference_date,
        ),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def aggregate_query_results(
    query_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not query_results:
        return {
            "query_count": 0,
            **{key: 0.0 for key in METRIC_KEYS},
            "latency_ms": 0.0,
            "latency_ms_p50": 0.0,
            "latency_ms_p95": 0.0,
        }

    metrics = {
        key: round(
            mean(
                float((result.get("metrics") or {}).get(key, 0.0))
                for result in query_results
            ),
            6,
        )
        for key in METRIC_KEYS
    }
    latencies = [float(result.get("latency_ms") or 0.0) for result in query_results]
    return {
        "query_count": len(query_results),
        **metrics,
        "latency_ms": round(mean(latencies), 3),
        "latency_ms_p50": round(median(latencies), 3),
        "latency_ms_p95": round(_percentile(latencies, 0.95), 3),
    }


def metric_delta(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, float]:
    keys = (*METRIC_KEYS, "latency_ms", "latency_ms_p50", "latency_ms_p95")
    return {
        key: round(
            float(candidate.get(key, 0.0)) - float(baseline.get(key, 0.0)),
            6,
        )
        for key in keys
    }


def report_contract_failures(
    report: Mapping[str, Any],
    *,
    min_headline_precision_at_5: float | None = None,
    min_headline_hit_at_5: float | None = None,
    max_headline_inactive_exposure: float = 0.0,
) -> list[str]:
    """Validate the evaluation contract without rerunning model inference.

    The natural-language system is the only headline candidate. The explicit
    structured-query system is retained solely as an oracle upper bound, while
    raw unfiltered CLIP remains an inactive-notice diagnostic.
    """

    failures: list[str] = []
    contract = report.get("evaluation_contract")
    systems = report.get("systems")
    if not isinstance(contract, Mapping):
        return ["evaluation_contract is missing"]
    if not isinstance(systems, Mapping):
        return ["systems is missing"]

    expected_roles = {
        "headline_system": HEADLINE_SYSTEM_ID,
        "baseline_system": BASELINE_SYSTEM_ID,
        "oracle_system": ORACLE_SYSTEM_ID,
        "diagnostic_system": DIAGNOSTIC_SYSTEM_ID,
    }
    for field, expected in expected_roles.items():
        actual = clean_text(contract.get(field))
        if actual != expected:
            failures.append(f"{field} must be {expected!r}, got {actual!r}")
        if expected not in systems:
            failures.append(f"required system is missing: {expected}")

    headline = systems.get(HEADLINE_SYSTEM_ID)
    baseline = systems.get(BASELINE_SYSTEM_ID)
    oracle = systems.get(ORACLE_SYSTEM_ID)
    diagnostic = systems.get(DIAGNOSTIC_SYSTEM_ID)
    if not all(
        isinstance(item, Mapping) for item in (headline, baseline, oracle, diagnostic)
    ):
        return failures

    if bool(headline.get("uses_explicit_structured")):
        failures.append("headline system must not use explicit structured query input")
    if not bool(oracle.get("uses_explicit_structured")):
        failures.append("oracle system must record explicit structured query use")
    if bool(diagnostic.get("status_filtered")):
        failures.append("raw diagnostic system must remain unfiltered")

    qrels = report.get("qrels")
    expected_query_count = len(qrels) if isinstance(qrels, Sequence) else 0
    if expected_query_count <= 0:
        failures.append("qrels must contain at least one query")

    status_filtered_ids = contract.get("status_filtered_systems")
    if not isinstance(status_filtered_ids, Sequence) or isinstance(
        status_filtered_ids, (str, bytes, bytearray)
    ):
        failures.append("status_filtered_systems must be an array")
        status_filtered_ids = ()
    for system_id in status_filtered_ids:
        system = systems.get(clean_text(system_id))
        if not isinstance(system, Mapping):
            failures.append(f"status-filtered system is missing: {system_id}")
            continue
        metrics = system.get("metrics")
        queries = system.get("queries")
        if not isinstance(metrics, Mapping) or not isinstance(queries, Sequence):
            failures.append(f"system output is incomplete: {system_id}")
            continue
        if expected_query_count and len(queries) != expected_query_count:
            failures.append(
                f"{system_id} query count {len(queries)} != qrels "
                f"{expected_query_count}"
            )
        inactive = float(metrics.get("inactive_exposure") or 0.0)
        if inactive > 1e-9:
            failures.append(f"{system_id} exposes inactive notices: {inactive:.6f}")

    if isinstance(headline, Mapping):
        headline_metrics = headline.get("metrics")
        if isinstance(headline_metrics, Mapping):
            precision = float(headline_metrics.get("precision@5") or 0.0)
            hit = float(headline_metrics.get("hit@5") or 0.0)
            inactive = float(headline_metrics.get("inactive_exposure") or 0.0)
            if (
                min_headline_precision_at_5 is not None
                and precision < min_headline_precision_at_5
            ):
                failures.append(
                    "headline precision@5 "
                    f"{precision:.6f} < {min_headline_precision_at_5:.6f}"
                )
            if min_headline_hit_at_5 is not None and hit < min_headline_hit_at_5:
                failures.append(
                    f"headline hit@5 {hit:.6f} < {min_headline_hit_at_5:.6f}"
                )
            if inactive > max_headline_inactive_exposure:
                failures.append(
                    "headline inactive_exposure "
                    f"{inactive:.6f} > {max_headline_inactive_exposure:.6f}"
                )
        else:
            failures.append("headline metrics are missing")

        for row in headline.get("queries") or []:
            retrieval = row.get("retrieval") if isinstance(row, Mapping) else None
            if not isinstance(retrieval, Mapping):
                failures.append("headline query retrieval metadata is missing")
                continue
            if bool(retrieval.get("uses_explicit_structured")):
                failures.append(
                    f"headline query {clean_text(row.get('id'))!r} used "
                    "explicit structured input"
                )
            if clean_text(retrieval.get("structured_source")) != "natural":
                failures.append(
                    f"headline query {clean_text(row.get('id'))!r} did not use "
                    "the natural parser source"
                )
            if retrieval.get("structured_query") != retrieval.get(
                "natural_parsed_query"
            ):
                failures.append(
                    f"headline query {clean_text(row.get('id'))!r} structured "
                    "conditions differ from its natural parser output"
                )
    return failures


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path, **extra: Any) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": stat.st_size,
        **extra,
    }


def qrels_record(
    queries: Sequence[Mapping[str, Any]],
    qrels: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    return [
        {
            "id": clean_text(query.get("id")),
            "criteria": dict(query.get("criteria") or {}),
            "relevant_count": len(qrels.get(clean_text(query.get("id")), set())),
            "relevant_ids": sorted(qrels.get(clean_text(query.get("id")), set())),
        }
        for query in queries
    ]


def _metric_cell(value: Any, percent: bool = False) -> str:
    number = float(value or 0.0)
    return f"{number * 100:.2f}%" if percent else f"{number:.3f}"


def report_to_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact, auditable Markdown companion to the JSON report."""

    systems = report.get("systems") or {}
    contract = report.get("evaluation_contract") or {}
    baseline_id = clean_text(contract.get("baseline_system")) or "clip_only_raw"
    headline_id = (
        clean_text(contract.get("headline_system")) or "hybrid_graph_structured"
    )
    oracle_id = clean_text(contract.get("oracle_system"))
    diagnostic_id = clean_text(contract.get("diagnostic_system"))
    baseline = systems.get(baseline_id) or {}
    candidate = systems.get(headline_id) or {}
    baseline_metrics = baseline.get("metrics") or {}
    candidate_metrics = candidate.get("metrics") or {}
    delta = report.get("delta") or {}
    artifacts = report.get("artifacts") or {}
    corpus = report.get("corpus") or {}

    lines = [
        "# 외형 검색품질 평가",
        "",
        f"- 기준일: `{clean_text(report.get('reference_date'))}`",
        f"- 쿼리 수: {baseline_metrics.get('query_count', 0)}",
        (
            "- Silver qrels: 공공 공고의 `color`, `weight`, `age`, `region`, "
            "`notice status`만 사용"
        ),
        "- 검색 결과 텍스트·순위·설명·VLM 출력은 정답 라벨 생성에 사용하지 않음",
        (
            "- `generated_at`은 실행 시각이라 매 실행마다 달라지며, latency는 "
            "하드웨어·워밍업 상태에 따라 달라질 수 있음"
        ),
        (
            "- inactive exposure: 기준일에 종료/만료된 공고가 "
            "상위 10개 결과에서 차지하는 비율"
        ),
        (
            f"- 대표 시스템: `{headline_id}` — 자연어 파싱 결과만 사용하고 "
            "평가 JSON의 `structured` 힌트는 사용하지 않음"
        ),
    ]
    if oracle_id:
        lines.append(
            f"- `{oracle_id}`는 명시적 구조 조건을 주입한 상한선이며 대표 성능이 아님"
        )
    if diagnostic_id:
        lines.append(
            f"- `{diagnostic_id}`는 종료 공고 노출 진단용이며 성능 baseline이 아님"
        )
    lines.extend(
        [
            "",
            "## 데이터 고정 정보",
            "",
            "| 항목 | 경로 | SHA-256 |",
            "|---|---|---|",
        ]
    )
    for name in ("index", "metas", "queries"):
        record = artifacts.get(name) or {}
        digest = clean_text(record.get("sha256"))
        lines.append(
            f"| {name} | `{clean_text(record.get('path'))}` | `{digest[:16]}…` |"
        )
    lines.extend(
        [
            "",
            (
                f"문서 {corpus.get('documents', 0)}건 중 기준일 활성 "
                f"{corpus.get('active_documents', 0)}건을 qrels 대상으로 사용했습니다."
            ),
            "",
            "## 전체 지표",
            "",
            f"| 지표 | {baseline_id} | {headline_id} | Δ |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in METRIC_KEYS:
        percent = key != "MRR"
        lines.append(
            f"| {key} | {_metric_cell(baseline_metrics.get(key), percent)} | "
            f"{_metric_cell(candidate_metrics.get(key), percent)} | "
            f"{_metric_cell(delta.get(key), percent)} |"
        )
    lines.append(
        "| latency (mean ms) | "
        f"{_metric_cell(baseline_metrics.get('latency_ms'))} | "
        f"{_metric_cell(candidate_metrics.get('latency_ms'))} | "
        f"{_metric_cell(delta.get('latency_ms'))} |"
    )
    lines.extend(
        [
            "",
            "## 단계별 비교",
            "",
            (
                "| 시스템 | 역할 | 명시적 structured 주입 | 상태 필터 | "
                "P@5 | Recall@10 | nDCG@5 | Hit@5 | inactive |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    system_order = contract.get("system_order") or list(systems)
    for system_id in system_order:
        system = systems.get(system_id) or {}
        metrics = system.get("metrics") or {}
        lines.append(
            f"| `{clean_text(system_id)}` | {clean_text(system.get('role'))} | "
            f"{'예' if system.get('uses_explicit_structured') else '아니오'} | "
            f"{'예' if system.get('status_filtered') else '아니오'} | "
            f"{_metric_cell(metrics.get('precision@5'), True)} | "
            f"{_metric_cell(metrics.get('recall@10'), True)} | "
            f"{_metric_cell(metrics.get('nDCG@5'), True)} | "
            f"{_metric_cell(metrics.get('hit@5'), True)} | "
            f"{_metric_cell(metrics.get('inactive_exposure'), True)} |"
        )
    lines.extend(
        [
            "",
            "## 쿼리별 결과",
            "",
            (
                f"| ID | 관련 공고 수 | {baseline_id} P@5 | {headline_id} P@5 | "
                f"{baseline_id} 상위 ID | {headline_id} 상위 ID |"
            ),
            "|---|---:|---:|---:|---|---|",
        ]
    )
    candidate_by_id = {
        clean_text(row.get("id")): row for row in candidate.get("queries") or []
    }
    for row in baseline.get("queries") or []:
        query_id = clean_text(row.get("id"))
        other = candidate_by_id.get(query_id, {})
        baseline_top = ", ".join(row.get("top_ids") or [])
        candidate_top = ", ".join(other.get("top_ids") or [])
        lines.append(
            f"| {query_id} | {row.get('relevant_count', 0)} | "
            f"{_metric_cell((row.get('metrics') or {}).get('precision@5'), True)} | "
            f"{_metric_cell((other.get('metrics') or {}).get('precision@5'), True)} | "
            f"`{baseline_top}` | `{candidate_top}` |"
        )
    lines.extend(
        [
            "",
            "## 해석 주의사항",
            "",
            (
                "이 평가는 공고에 명시된 객관 필드로 만든 silver label을 사용합니다. "
                "입양 적합성·성격·사회성을 평가하지 않으며, 실제 관련성의 완전한 "
                "정답으로 간주할 수 없습니다."
            ),
            "",
            (
                "공고 필드가 누락되거나 잘못 기재된 후보는 관련 공고에서 빠질 수 있고, "
                "색상·체중 구간·연령·광역 지역이 같아도 사용자의 실제 시각적 의도와 "
                "다를 수 있습니다. 따라서 수치는 사람 검수 정답이 아닌 시스템 간 "
                "회귀 비교용 근사치입니다."
            ),
            "",
            (
                "재현 시 `reference_date`와 세 입력 파일의 SHA-256을 고정해야 합니다. "
                "JSON의 `generated_at`과 latency 값은 결정적 산출물이 아닙니다."
            ),
            "",
            (
                "대표 수치는 자연어 end-to-end 회귀값입니다. `structured_oracle`은 "
                "파서 정답을 주입한 상한선이므로 대표 수치나 자연어 이해 성능으로 "
                "인용하면 안 됩니다."
            ),
            "",
        ]
    )
    return "\n".join(lines)
