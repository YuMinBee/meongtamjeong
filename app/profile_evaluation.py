from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from app.profile_rerank import (
    APPEARANCE_CONDITION_KEYS,
    ProfileRerankSettings,
    UserProfile,
    rerank_candidates,
)


APPEARANCE_FIELDS = ("preferred_size", "preferred_age", "preferred_region")
_UNKNOWN_TEXT = {
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
_WEIGHT_RE = re.compile(r"([0-9]+(?:[.,][0-9]+)?)")
_YEAR_RE = re.compile(r"(?:19|20)[0-9]{2}")
_AGE_RE = re.compile(r"([0-9]+(?:[.,][0-9]+)?)\s*(?:살|세)")
_MONTH_RE = re.compile(r"([0-9]+(?:[.,][0-9]+)?)\s*개월")
_REGION_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("서울", ("서울",)),
    (
        "경기",
        (
            "경기",
            "수원",
            "성남",
            "고양",
            "용인",
            "부천",
            "안산",
            "안양",
            "평택",
            "의정부",
            "시흥",
            "화성",
            "김포",
            "파주",
            "광명",
            "군포",
            "하남",
            "이천",
            "안성",
            "오산",
            "양주",
            "포천",
            "여주",
            "가평",
            "양평",
        ),
    ),
    ("인천", ("인천",)),
    ("강원", ("강원", "춘천", "원주", "강릉", "동해", "태백", "속초", "삼척")),
    ("충북", ("충북", "청주", "충주", "제천", "음성", "진천")),
    ("충남", ("충남", "천안", "공주", "아산", "논산", "보령", "서산", "당진", "홍성")),
    ("대전", ("대전",)),
    ("세종", ("세종",)),
    ("전북", ("전북", "전주", "군산", "익산", "정읍", "남원", "김제")),
    ("전남", ("전남", "목포", "여수", "순천", "나주", "광양")),
    ("광주", ("광주",)),
    (
        "경북",
        (
            "경북",
            "포항",
            "경주",
            "김천",
            "안동",
            "구미",
            "영주",
            "영천",
            "상주",
            "문경",
        ),
    ),
    ("경남", ("경남", "창원", "진주", "통영", "사천", "김해", "밀양", "거제", "양산")),
    ("대구", ("대구",)),
    ("울산", ("울산",)),
    ("부산", ("부산",)),
    ("제주", ("제주",)),
)
_ROW_TYPE_PRIORITY = {"text": 0, "image": 1, "crop_image": 2}
_CERTAINTY_PATTERNS = (
    re.compile(r"입양(?:에|하기에)?\s*적합(?:합니다|하다|한|함)"),
    re.compile(r"입양\s*적합도(?:가)?\s*(?:높|확실)"),
    re.compile(r"반드시\s*입양"),
    re.compile(r"완벽한\s*(?:입양|반려)"),
    re.compile(r"확실히\s*적합"),
    re.compile(r"최적의\s*입양"),
    re.compile(r"입양을\s*추천합니다"),
)


def _known(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        return any(_known(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_known(item) for item in value)
    return str(value).strip().casefold() not in _UNKNOWN_TEXT


def _notice_id(row: Mapping[str, Any]) -> str:
    for key in ("desertionNo", "desertion_no", "dog_id"):
        if _known(row.get(key)):
            return str(row[key]).strip()
    return ""


def join_notice_rows(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Join image/crop/text rows to one deterministic record per notice ID."""

    grouped: Dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for index, row in enumerate(rows):
        notice_id = _notice_id(row)
        if notice_id:
            grouped.setdefault(notice_id, []).append((index, row))

    joined: Dict[str, Dict[str, Any]] = {}
    for notice_id in sorted(grouped):
        ordered = sorted(
            grouped[notice_id],
            key=lambda item: (
                _ROW_TYPE_PRIORITY.get(str(item[1].get("type", "")), 99),
                item[0],
            ),
        )
        merged: Dict[str, Any] = {}
        for _, row in ordered:
            for key, value in row.items():
                if key not in merged or not _known(merged[key]):
                    merged[key] = deepcopy(value)
        merged["desertionNo"] = notice_id
        joined[notice_id] = merged
    return joined


def load_joined_metas(path: Path) -> tuple[Dict[str, Dict[str, Any]], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or any(
        not isinstance(row, dict) for row in payload
    ):
        raise ValueError("dog_metas JSON must be an array of objects")
    return join_notice_rows(payload), len(payload)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def portable_input_path(path: Path) -> str:
    resolved = path.resolve()
    for parent in (resolved.parent, *resolved.parents):
        if (parent / ".git").exists():
            return resolved.relative_to(parent).as_posix()
    return resolved.name


def _raw_weight_size(value: Any) -> tuple[str, float | None]:
    match = _WEIGHT_RE.search(str(value or "").replace(" ", ""))
    if not match:
        return "unknown", None
    weight = float(match.group(1).replace(",", "."))
    if not math.isfinite(weight) or weight <= 0:
        return "unknown", None
    if weight <= 8:
        return "small", weight
    if weight <= 18:
        return "medium", weight
    return "large", weight


def _raw_age_group(value: Any, reference_year: int) -> tuple[str, float | None]:
    text = str(value or "").strip()
    year_match = _YEAR_RE.search(text)
    if year_match:
        years = reference_year - int(year_match.group(0))
        if years < 0:
            return "unknown", None
        if years <= 1:
            return "puppy", float(years)
        if years >= 8:
            return "senior", float(years)
        return "adult", float(years)
    age_match = _AGE_RE.search(text)
    if age_match:
        years = float(age_match.group(1).replace(",", "."))
    else:
        month_match = _MONTH_RE.search(text)
        if month_match:
            years = float(month_match.group(1).replace(",", ".")) / 12.0
        elif any(token in text.casefold() for token in ("puppy", "새끼")):
            return "puppy", None
        elif any(token in text.casefold() for token in ("senior", "노령", "노견")):
            return "senior", None
        elif any(token in text.casefold() for token in ("adult", "성견")):
            return "adult", None
        else:
            return "unknown", None
    if years < 0:
        return "unknown", None
    if years <= 1:
        return "puppy", years
    if years >= 8:
        return "senior", years
    return "adult", years


def _raw_region(meta: Mapping[str, Any]) -> tuple[str, str]:
    source_keys = (
        "region",
        "org_name",
        "orgNm",
        "care_addr",
        "careAddr",
        "happen_place",
        "happenPlace",
        "care_name",
        "careNm",
    )
    source_values = [
        str(meta[key]).strip() for key in source_keys if _known(meta.get(key))
    ]
    haystack = " ".join(source_values)
    for region, hints in _REGION_HINTS:
        if any(hint in haystack for hint in hints):
            return region, haystack
    return "unknown", haystack


def raw_appearance(meta: Mapping[str, Any], reference_year: int) -> Dict[str, Any]:
    """Extract silver-label fields from source notice values, never VLM fields."""

    size, weight_kg = _raw_weight_size(meta.get("weight"))
    age_group, age_years = _raw_age_group(meta.get("age"), reference_year)
    region, region_source = _raw_region(meta)
    return {
        "size": size,
        "age_group": age_group,
        "region": region,
        "source_values": {
            "weight": meta.get("weight"),
            "age": meta.get("age"),
            "region_text": region_source,
        },
        "parsed_values": {
            "weight_kg": weight_kg,
            "age_years": age_years,
        },
    }


def derive_silver_label(
    meta: Mapping[str, Any],
    profile: UserProfile,
    *,
    reference_year: int,
) -> Dict[str, Any]:
    appearance = raw_appearance(meta, reference_year)
    desired = {
        "preferred_size": profile.preferred_size.value,
        "preferred_age": profile.preferred_age.value,
        "preferred_region": profile.preferred_region,
    }
    actual = {
        "preferred_size": appearance["size"],
        "preferred_age": appearance["age_group"],
        "preferred_region": appearance["region"],
    }
    states: Dict[str, str] = {}
    for key in APPEARANCE_FIELDS:
        wanted = desired[key]
        if wanted in (None, "", "any"):
            continue
        observed = actual[key]
        if observed == "unknown":
            states[key] = "unknown"
        elif key == "preferred_region":
            wanted_key = re.sub(r"\s+", "", str(wanted)).casefold()
            observed_key = re.sub(r"\s+", "", str(observed)).casefold()
            states[key] = (
                "matched"
                if wanted_key in observed_key or observed_key in wanted_key
                else "mismatch"
            )
        else:
            states[key] = "matched" if wanted == observed else "mismatch"

    applicable_count = len(states)
    match_count = sum(state == "matched" for state in states.values())
    known_count = sum(state != "unknown" for state in states.values())
    grade = match_count / applicable_count if applicable_count else 0.0
    return {
        "relevance_grade": round(grade, 6),
        "binary_relevant": bool(states)
        and all(state == "matched" for state in states.values()),
        "match_count": match_count,
        "known_count": known_count,
        "applicable_count": applicable_count,
        "condition_states": states,
        "appearance": appearance,
    }


def ranking_metrics(
    order: Sequence[str],
    silver_labels: Mapping[str, Mapping[str, Any]],
    *,
    k: int = 5,
) -> Dict[str, float]:
    cutoff = min(k, len(order))
    top = list(order[:cutoff])
    relevant = [bool(silver_labels[item]["binary_relevant"]) for item in top]
    precision = sum(relevant) / cutoff if cutoff else 0.0
    first_relevant = next(
        (index + 1 for index, value in enumerate(relevant) if value), None
    )
    mrr = 1.0 / first_relevant if first_relevant else 0.0

    gains = [float(silver_labels[item]["relevance_grade"]) for item in top]
    dcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_gains = sorted(
        (float(label["relevance_grade"]) for label in silver_labels.values()),
        reverse=True,
    )[:cutoff]
    idcg = sum(
        (2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(ideal_gains)
    )
    return {
        f"precision_at_{k}": round(precision, 6),
        f"ndcg_at_{k}": round(dcg / idcg if idcg else 0.0, 6),
        "mrr": round(mrr, 6),
    }


def rank_change(baseline: Sequence[str], reranked: Sequence[str]) -> Dict[str, Any]:
    baseline_rank = {notice_id: index + 1 for index, notice_id in enumerate(baseline)}
    reranked_rank = {notice_id: index + 1 for index, notice_id in enumerate(reranked)}
    shifts = {
        notice_id: baseline_rank[notice_id] - reranked_rank[notice_id]
        for notice_id in baseline
    }
    absolute = [abs(value) for value in shifts.values()]
    return {
        "changed_candidates": sum(value != 0 for value in shifts.values()),
        "mean_absolute_rank_change": round(sum(absolute) / len(absolute), 6)
        if absolute
        else 0.0,
        "max_absolute_rank_change": max(absolute, default=0),
        "top1_changed": bool(baseline and reranked and baseline[0] != reranked[0]),
        "shifts": shifts,
    }


def _evidence_consistency_failures(result: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    reason = str(result.get("recommendation_reason", ""))
    evidence_fields = (
        "matched_conditions",
        "caution_conditions",
        "unknown_conditions",
    )
    for field in evidence_fields:
        for message in result.get(field, []):
            if str(message) not in reason:
                failures.append(f"{result.get('dog_id')}:{field}:missing_from_reason")
    prefix = (
        f"확인된 조건 {result.get('evaluated_count')}/{result.get('applicable_count')}"
    )
    if prefix not in reason:
        failures.append(f"{result.get('dog_id')}:count_summary_missing")
    groups = [set(result.get(field, [])) for field in evidence_fields]
    if (groups[0] & groups[1]) or (groups[0] & groups[2]) or (groups[1] & groups[2]):
        failures.append(f"{result.get('dog_id')}:evidence_groups_overlap")
    return failures


def _certainty_hits(result: Mapping[str, Any]) -> list[str]:
    reason = str(result.get("recommendation_reason", ""))
    return [
        pattern.pattern for pattern in _CERTAINTY_PATTERNS if pattern.search(reason)
    ]


def _unknown_neutrality_probe(
    profile: UserProfile,
    settings: ProfileRerankSettings,
    reference_date: datetime,
) -> Dict[str, Any]:
    retrieval_score = 0.5
    result = rerank_candidates(
        [
            {
                "score": retrieval_score,
                "desertionNo": "unknown-neutrality-probe",
                "process_state": "보호중",
                "notice_end": f"{reference_date.year + 1}1231",
            }
        ],
        profile,
        settings,
        topk=1,
        include_unknown_notices=False,
        reference_date=reference_date,
        condition_keys=APPEARANCE_CONDITION_KEYS,
    )[0]
    delta = float(result["final_score"]) - float(result["retrieval_score"])
    return {
        "passed": abs(delta) <= 1e-9 and result["compatibility_score"] == 0,
        "retrieval_score": result["retrieval_score"],
        "compatibility_score": result["compatibility_score"],
        "final_score": result["final_score"],
        "score_delta": round(delta, 9),
        "unknown_condition_count": len(result["unknown_conditions"]),
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    candidate_ids = config.get("candidate_ids")
    profiles = config.get("profiles")
    if not isinstance(candidate_ids, list) or len(candidate_ids) < 5:
        raise ValueError("candidate_ids must contain at least five fixed notice IDs")
    if len(candidate_ids) != len(set(map(str, candidate_ids))):
        raise ValueError("candidate_ids must be unique")
    if not isinstance(profiles, list) or len(profiles) < 2:
        raise ValueError("at least two profiles are required")
    ids = [str(item.get("id", "")) for item in profiles if isinstance(item, dict)]
    if (
        len(ids) != len(profiles)
        or len(ids) != len(set(ids))
        or any(not item for item in ids)
    ):
        raise ValueError("profile IDs must be present and unique")

    sensitivity = config.get("quality_weight_sensitivity")
    if sensitivity is not None and not isinstance(sensitivity, Mapping):
        raise ValueError("quality_weight_sensitivity must be an object")


def _quality_sensitivity_settings(
    config: Mapping[str, Any],
    primary_settings: ProfileRerankSettings,
    candidate_count: int,
) -> Dict[str, Any]:
    payload = config.get("quality_weight_sensitivity") or {}
    if not isinstance(payload, Mapping):
        raise ValueError("quality_weight_sensitivity must be an object")

    def parse_weight(key: str, default: float) -> float:
        raw = payload.get(key, default)
        if isinstance(raw, bool):
            raise ValueError(f"quality_weight_sensitivity.{key} must be a number")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"quality_weight_sensitivity.{key} must be a number"
            ) from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"quality_weight_sensitivity.{key} must be finite and >= 0"
            )
        return value

    code_default = float(ProfileRerankSettings().quality_weight)
    control = parse_weight(
        "control_quality_weight", float(primary_settings.quality_weight)
    )
    production_default = parse_weight("production_default_quality_weight", code_default)
    raw_top_k = payload.get("top_k", 5)
    if isinstance(raw_top_k, bool):
        raise ValueError("quality_weight_sensitivity.top_k must be an integer")
    try:
        top_k = int(raw_top_k)
    except (TypeError, ValueError) as exc:
        raise ValueError("quality_weight_sensitivity.top_k must be an integer") from exc
    if top_k < 1 or top_k > candidate_count:
        raise ValueError(
            "quality_weight_sensitivity.top_k must be between 1 and candidate count"
        )
    return {
        "control_quality_weight": control,
        "production_default_quality_weight": production_default,
        "code_default_quality_weight": code_default,
        "top_k": top_k,
    }


def _rank_score_context(
    order: Sequence[str],
    results_by_id: Mapping[str, Mapping[str, Any]],
    notice_id: str,
) -> Dict[str, Any]:
    rank = order.index(notice_id) + 1
    score = float(results_by_id[notice_id]["final_score"])
    higher_id = order[rank - 2] if rank > 1 else None
    lower_id = order[rank] if rank < len(order) else None
    margin_to_higher = (
        round(float(results_by_id[higher_id]["final_score"]) - score, 6)
        if higher_id is not None
        else None
    )
    margin_over_lower = (
        round(score - float(results_by_id[lower_id]["final_score"]), 6)
        if lower_id is not None
        else None
    )
    return {
        "rank": rank,
        "final_score": round(score, 6),
        "higher_notice_id": higher_id,
        "margin_to_higher": margin_to_higher,
        "lower_notice_id": lower_id,
        "margin_over_lower": margin_over_lower,
    }


def _optional_number_changed(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left != right
    return abs(float(left) - float(right)) > 1e-9


def _quality_weight_sensitivity(
    config: Mapping[str, Any],
    candidates: Sequence[Dict[str, Any]],
    candidate_ids: Sequence[str],
    profile_specs: Sequence[Mapping[str, Any]],
    primary_profile_reports: Sequence[Mapping[str, Any]],
    reference_date: datetime,
    primary_settings: ProfileRerankSettings,
) -> tuple[Dict[str, Any], list[str]]:
    sensitivity_settings = _quality_sensitivity_settings(
        config, primary_settings, len(candidate_ids)
    )
    control_weight = float(sensitivity_settings["control_quality_weight"])
    production_weight = float(sensitivity_settings["production_default_quality_weight"])
    code_default_weight = float(sensitivity_settings["code_default_quality_weight"])
    top_k = int(sensitivity_settings["top_k"])
    failures: list[str] = []
    if abs(float(primary_settings.quality_weight)) > 1e-9:
        failures.append("primary_contract_quality_weight_not_zero")
    if abs(control_weight - float(primary_settings.quality_weight)) > 1e-9:
        failures.append("control_weight_differs_from_primary_contract")
    if abs(production_weight - code_default_weight) > 1e-9:
        failures.append("production_default_weight_differs_from_code_default")

    primary_orders = {
        str(profile["profile_id"]): list(profile["reranked_order"])
        for profile in primary_profile_reports
    }
    quality_by_id: Dict[str, float | None] = {}
    profile_reports: list[Dict[str, Any]] = []
    unknown_observations: list[Dict[str, Any]] = []

    for profile_spec in profile_specs:
        profile_id = str(profile_spec["id"])
        profile = UserProfile.model_validate(profile_spec["profile"])
        runs: Dict[str, list[Dict[str, Any]]] = {}
        for run_name, quality_weight in (
            ("control", control_weight),
            ("production_default", production_weight),
        ):
            run_settings = ProfileRerankSettings(
                compatibility_weight=float(primary_settings.compatibility_weight),
                quality_weight=quality_weight,
                candidate_multiplier=1,
            )
            results = rerank_candidates(
                candidates,
                profile,
                run_settings,
                topk=len(candidates),
                include_unknown_notices=False,
                reference_date=reference_date,
                condition_keys=APPEARANCE_CONDITION_KEYS,
            )
            order = [str(result["dog_id"]) for result in results]
            if set(order) != set(candidate_ids):
                failures.append(f"{profile_id}:{run_name}:candidate_population_changed")
            runs[run_name] = results

        control_results = runs["control"]
        production_results = runs["production_default"]
        control_order = [str(result["dog_id"]) for result in control_results]
        production_order = [str(result["dog_id"]) for result in production_results]
        if control_order != primary_orders.get(profile_id):
            failures.append(f"{profile_id}:control_order_differs_from_primary_contract")

        control_by_id = {str(result["dog_id"]): result for result in control_results}
        production_by_id = {
            str(result["dog_id"]): result for result in production_results
        }
        for notice_id in candidate_ids:
            control_result = control_by_id[notice_id]
            production_result = production_by_id[notice_id]
            if (
                control_result["compatibility_score"]
                != production_result["compatibility_score"]
            ):
                failures.append(
                    f"{profile_id}:{notice_id}:compatibility_changed_with_quality_weight"
                )
            control_quality = control_result["quality_score"]
            production_quality = production_result["quality_score"]
            if control_quality != production_quality:
                failures.append(
                    f"{profile_id}:{notice_id}:quality_value_changed_between_runs"
                )
            if (
                notice_id in quality_by_id
                and quality_by_id[notice_id] != control_quality
            ):
                failures.append(
                    f"{profile_id}:{notice_id}:quality_value_changed_between_profiles"
                )
            quality_by_id.setdefault(notice_id, control_quality)

        control_top = control_order[:top_k]
        production_top = production_order[:top_k]
        control_top_set = set(control_top)
        production_top_set = set(production_top)
        entered = [
            notice_id
            for notice_id in production_top
            if notice_id not in control_top_set
        ]
        exited = [
            notice_id
            for notice_id in control_top
            if notice_id not in production_top_set
        ]
        changed_slots = [
            {
                "rank": index + 1,
                "control_notice_id": control_notice_id,
                "production_default_notice_id": production_notice_id,
            }
            for index, (control_notice_id, production_notice_id) in enumerate(
                zip(control_top, production_top)
            )
            if control_notice_id != production_notice_id
        ]
        changed_position_ids = [
            notice_id
            for notice_id in control_top
            if notice_id in production_top_set
            and control_top.index(notice_id) != production_top.index(notice_id)
        ]

        unknown_rows: list[Dict[str, Any]] = []
        for notice_id in candidate_ids:
            if control_by_id[notice_id]["quality_score"] is not None:
                continue
            control_context = _rank_score_context(
                control_order, control_by_id, notice_id
            )
            production_context = _rank_score_context(
                production_order, production_by_id, notice_id
            )
            score_delta = round(
                float(production_context["final_score"])
                - float(control_context["final_score"]),
                6,
            )
            direct_score_neutral = abs(score_delta) <= 1e-9
            relative_context_changed = (
                control_context["higher_notice_id"]
                != production_context["higher_notice_id"]
                or control_context["lower_notice_id"]
                != production_context["lower_notice_id"]
                or _optional_number_changed(
                    control_context["margin_to_higher"],
                    production_context["margin_to_higher"],
                )
                or _optional_number_changed(
                    control_context["margin_over_lower"],
                    production_context["margin_over_lower"],
                )
            )
            row = {
                "notice_id": notice_id,
                "quality_score": None,
                "control": control_context,
                "production_default": production_context,
                "rank_shift": int(control_context["rank"])
                - int(production_context["rank"]),
                "score_delta": score_delta,
                "direct_score_neutral": direct_score_neutral,
                "relative_neighbor_or_margin_changed": relative_context_changed,
            }
            if not direct_score_neutral:
                failures.append(
                    f"{profile_id}:{notice_id}:unknown_quality_score_changed"
                )
            unknown_rows.append(row)
            unknown_observations.append({"profile_id": profile_id, **row})

        profile_reports.append(
            {
                "profile_id": profile_id,
                "label": str(profile_spec.get("label", profile_id)),
                "control_order": control_order,
                "production_default_order": production_order,
                "top1": {
                    "control_notice_id": control_order[0],
                    "production_default_notice_id": production_order[0],
                    "changed": control_order[0] != production_order[0],
                },
                "top_k_membership": {
                    "k": top_k,
                    "control_notice_ids": control_top,
                    "production_default_notice_ids": production_top,
                    "entered_notice_ids": entered,
                    "exited_notice_ids": exited,
                    "replacement_count": len(entered),
                    "changed_candidate_count": len(entered) + len(exited),
                },
                "top_k_position": {
                    "changed_candidate_count": len(changed_position_ids),
                    "changed_notice_ids": changed_position_ids,
                    "changed_slot_count": len(changed_slots),
                    "changed_slots": changed_slots,
                },
                "all_candidate_rank_change": rank_change(
                    control_order, production_order
                ),
                "unknown_quality_candidates": unknown_rows,
            }
        )

    known_ids = [
        notice_id for notice_id in candidate_ids if quality_by_id[notice_id] is not None
    ]
    unknown_ids = [
        notice_id for notice_id in candidate_ids if quality_by_id[notice_id] is None
    ]
    known_values = [float(quality_by_id[notice_id]) for notice_id in known_ids]

    def stats(values: Sequence[float]) -> Dict[str, float | None]:
        if not values:
            return {"min": None, "mean": None, "max": None}
        return {
            "min": round(min(values), 6),
            "mean": round(sum(values) / len(values), 6),
            "max": round(max(values), 6),
        }

    bonus_values = [value * production_weight for value in known_values]
    profile_mean_shifts = [
        float(profile["all_candidate_rank_change"]["mean_absolute_rank_change"])
        for profile in profile_reports
    ]
    profile_max_shifts = [
        int(profile["all_candidate_rank_change"]["max_absolute_rank_change"])
        for profile in profile_reports
    ]
    sensitivity_summary = {
        "passed": not failures,
        "failure_count": len(failures),
        "failure_details": failures,
        "profile_count": len(profile_reports),
        "profiles_with_top1_change": sum(
            bool(profile["top1"]["changed"]) for profile in profile_reports
        ),
        "profiles_with_top_k_membership_change": sum(
            bool(profile["top_k_membership"]["changed_candidate_count"])
            for profile in profile_reports
        ),
        "profiles_with_top_k_position_change": sum(
            bool(profile["top_k_position"]["changed_candidate_count"])
            for profile in profile_reports
        ),
        "mean_profile_mean_absolute_rank_shift": round(
            sum(profile_mean_shifts) / len(profile_mean_shifts), 6
        ),
        "max_absolute_rank_shift_across_profiles": max(profile_max_shifts, default=0),
        "unknown_profile_observation_count": len(unknown_observations),
        "unknown_observations_with_rank_change": sum(
            bool(row["rank_shift"]) for row in unknown_observations
        ),
        "unknown_observations_with_direct_score_change": sum(
            not bool(row["direct_score_neutral"]) for row in unknown_observations
        ),
        "unknown_observations_with_relative_context_change": sum(
            bool(row["relative_neighbor_or_margin_changed"])
            for row in unknown_observations
        ),
    }
    return (
        {
            "scope": "fixed_pool_quality_weight_sensitivity",
            "evaluation_type": "deterministic_sensitivity_contract_not_performance",
            "claim_boundary": (
                "Compares ordering sensitivity only; it does not show search "
                "quality, user utility, or an optimal weight."
            ),
            "settings": {
                **sensitivity_settings,
                "primary_contract_quality_weight": float(
                    primary_settings.quality_weight
                ),
                "compatibility_weight": float(primary_settings.compatibility_weight),
                "control_matches_primary_contract": abs(
                    control_weight - float(primary_settings.quality_weight)
                )
                <= 1e-9,
                "production_default_matches_code_default": abs(
                    production_weight - code_default_weight
                )
                <= 1e-9,
            },
            "quality_coverage": {
                "candidate_count": len(candidate_ids),
                "known_count": len(known_ids),
                "unknown_count": len(unknown_ids),
                "known_rate": round(len(known_ids) / len(candidate_ids), 6),
                "unknown_rate": round(len(unknown_ids) / len(candidate_ids), 6),
                "known_candidate_ids": known_ids,
                "unknown_candidate_ids": unknown_ids,
                "quality_scores": {
                    notice_id: quality_by_id[notice_id] for notice_id in candidate_ids
                },
                "known_quality_score_stats": stats(known_values),
                "known_quality_bonus_at_production_default_stats": stats(bonus_values),
            },
            "unknown_quality_policy": (
                "Unknown receives no direct quality term, while known candidates "
                "can gain a relative bonus; rank and neighboring score margins "
                "are therefore reported separately."
            ),
            "summary": sensitivity_summary,
            "profiles": profile_reports,
            "limitations": [
                "The 12 fixed candidates use deterministic synthetic retrieval scores and two configured profiles.",
                "Photo quality is a technical image-usability heuristic, not animal quality, adoption suitability, or human relevance judgment.",
                "Only one fixed candidate has unknown photo quality, so unknown relative effects do not generalize beyond this pool.",
            ],
        },
        failures,
    )


def evaluate_profile_reranking(
    config_path: Path,
    metas_path: Path,
) -> Dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("profile evaluation config must be an object")
    _validate_config(config)
    joined, metadata_rows = load_joined_metas(metas_path)

    candidate_ids = [str(value) for value in config["candidate_ids"]]
    missing = [notice_id for notice_id in candidate_ids if notice_id not in joined]
    if missing:
        raise ValueError("candidate IDs missing from dog_metas: " + ", ".join(missing))

    reference_date = datetime.fromisoformat(str(config["reference_date"]))
    score_start = float(config.get("baseline_score_start", 0.6))
    score_step = float(config.get("baseline_score_step", 0.002))
    settings_payload = config.get("settings", {})
    settings = ProfileRerankSettings(
        compatibility_weight=float(settings_payload.get("compatibility_weight", 0.25)),
        quality_weight=float(settings_payload.get("quality_weight", 0.0)),
        candidate_multiplier=1,
    )

    source_candidates = [joined[notice_id] for notice_id in candidate_ids]
    candidates = [
        {
            **deepcopy(meta),
            "score": score_start - index * score_step,
            "_raw_meta": deepcopy(meta),
        }
        for index, meta in enumerate(source_candidates)
    ]
    candidate_subset_hash = canonical_sha256(source_candidates)

    profile_reports: list[Dict[str, Any]] = []
    all_consistency_failures: list[str] = []
    all_certainty_hits: list[Dict[str, str]] = []
    for profile_spec in config["profiles"]:
        profile = UserProfile.model_validate(profile_spec["profile"])
        silver_labels = {
            notice_id: derive_silver_label(
                joined[notice_id],
                profile,
                reference_year=reference_date.year,
            )
            for notice_id in candidate_ids
        }
        reranked_results = rerank_candidates(
            candidates,
            profile,
            settings,
            topk=len(candidates),
            include_unknown_notices=False,
            reference_date=reference_date,
            condition_keys=APPEARANCE_CONDITION_KEYS,
        )
        reranked_order = [str(result["dog_id"]) for result in reranked_results]
        if set(reranked_order) != set(candidate_ids):
            absent = sorted(set(candidate_ids) - set(reranked_order))
            raise ValueError(
                "fixed candidates were filtered by notice status: " + ", ".join(absent)
            )

        by_id = {str(result["dog_id"]): result for result in reranked_results}
        consistency_failures: list[str] = []
        certainty_hits: list[Dict[str, str]] = []
        score_label_mismatches: list[str] = []
        for notice_id, result in by_id.items():
            consistency_failures.extend(_evidence_consistency_failures(result))
            certainty_hits.extend(
                {"notice_id": notice_id, "pattern": pattern}
                for pattern in _certainty_hits(result)
            )
            expected_score = float(silver_labels[notice_id]["relevance_grade"])
            if abs(float(result["compatibility_score"]) - expected_score) > 1e-6:
                score_label_mismatches.append(notice_id)

        baseline_metrics = ranking_metrics(candidate_ids, silver_labels, k=5)
        reranked_metrics = ranking_metrics(reranked_order, silver_labels, k=5)
        deltas = {
            key: round(reranked_metrics[key] - baseline_metrics[key], 6)
            for key in baseline_metrics
        }
        baseline_rank = {
            notice_id: index + 1 for index, notice_id in enumerate(candidate_ids)
        }
        reranked_rank = {
            notice_id: index + 1 for index, notice_id in enumerate(reranked_order)
        }
        candidate_rows = [
            {
                "notice_id": notice_id,
                "baseline_rank": baseline_rank[notice_id],
                "reranked_rank": reranked_rank[notice_id],
                "rank_change": baseline_rank[notice_id] - reranked_rank[notice_id],
                "retrieval_score": by_id[notice_id]["retrieval_score"],
                "compatibility_score": by_id[notice_id]["compatibility_score"],
                "final_score": by_id[notice_id]["final_score"],
                "silver_label": silver_labels[notice_id],
                "matched_conditions": by_id[notice_id]["matched_conditions"],
                "caution_conditions": by_id[notice_id]["caution_conditions"],
                "unknown_conditions": by_id[notice_id]["unknown_conditions"],
                "recommendation_reason": by_id[notice_id]["recommendation_reason"],
            }
            for notice_id in candidate_ids
        ]
        profile_report = {
            "profile_id": str(profile_spec["id"]),
            "label": str(profile_spec.get("label", profile_spec["id"])),
            "appearance_preferences": {
                key: profile.model_dump(mode="json")[key] for key in APPEARANCE_FIELDS
            },
            "unscored_lifestyle_fields": [
                "housing_type",
                "daily_absence_hours",
                "activity_level",
                "dog_experience",
                "has_children",
                "has_other_pets",
            ],
            "baseline_order": candidate_ids,
            "reranked_order": reranked_order,
            "baseline_metrics": baseline_metrics,
            "reranked_metrics": reranked_metrics,
            "metric_delta": deltas,
            "rank_change": rank_change(candidate_ids, reranked_order),
            "unknown_neutrality_probe": _unknown_neutrality_probe(
                profile, settings, reference_date
            ),
            "evidence_consistency_failures": consistency_failures,
            "silver_score_mismatch_notice_ids": score_label_mismatches,
            "adoption_suitability_certainty_hits": certainty_hits,
            "candidates": candidate_rows,
        }
        profile_reports.append(profile_report)
        all_consistency_failures.extend(
            f"{profile_spec['id']}:{failure}" for failure in consistency_failures
        )
        all_certainty_hits.extend(
            {"profile_id": str(profile_spec["id"]), **hit} for hit in certainty_hits
        )

    quality_sensitivity, quality_sensitivity_failures = _quality_weight_sensitivity(
        config,
        candidates,
        candidate_ids,
        config["profiles"],
        profile_reports,
        reference_date,
        settings,
    )

    precision_deltas = [
        profile["metric_delta"]["precision_at_5"] for profile in profile_reports
    ]
    ndcg_deltas = [profile["metric_delta"]["ndcg_at_5"] for profile in profile_reports]
    mrr_deltas = [profile["metric_delta"]["mrr"] for profile in profile_reports]
    top1_values = [profile["reranked_order"][0] for profile in profile_reports]
    neutrality_failures = [
        profile["profile_id"]
        for profile in profile_reports
        if not profile["unknown_neutrality_probe"]["passed"]
    ]
    score_mismatch_count = sum(
        len(profile["silver_score_mismatch_notice_ids"]) for profile in profile_reports
    )
    contract_failure_details = [
        *(f"unknown_neutrality:{profile_id}" for profile_id in neutrality_failures),
        *(f"evidence_consistency:{failure}" for failure in all_consistency_failures),
        *(
            f"silver_score_mismatch:{profile['profile_id']}:{notice_id}"
            for profile in profile_reports
            for notice_id in profile["silver_score_mismatch_notice_ids"]
        ),
        *(
            "adoption_suitability_certainty_phrase:"
            f"{hit['profile_id']}:{hit['notice_id']}:{hit['pattern']}"
            for hit in all_certainty_hits
        ),
        *(
            f"quality_weight_sensitivity:{failure}"
            for failure in quality_sensitivity_failures
        ),
    ]
    return {
        "schema_version": 2,
        "evaluation_id": str(config.get("evaluation_id", config_path.stem)),
        "evaluation_reference_date": str(config["reference_date"]),
        "scope": "appearance_profile_reranking",
        "method": {
            "evaluation_type": "deterministic_rerank_contract_not_performance",
            "candidate_pool": "same fixed notice IDs and baseline retrieval order for every profile",
            "production_function": "app.profile_rerank.rerank_candidates",
            "condition_keys": list(APPEARANCE_FIELDS),
            "silver_source_fields": ["weight", "age", "region/org/shelter address"],
            "silver_label": "exact public-notice field matches; unknown yields zero bonus and zero penalty",
            "binary_relevance": "all applicable appearance fields are known and match",
            "graded_relevance": "matched appearance fields divided by applicable appearance fields",
            "top_k": 5,
            "compatibility_weight": settings.compatibility_weight,
            "quality_weight": settings.quality_weight,
            "quality_contract_isolation": (
                "primary compatibility contract fixes quality_weight at 0.0"
            ),
            "lifestyle_or_behavior_scored": False,
        },
        "inputs": {
            "profile_config": portable_input_path(config_path),
            "profile_config_sha256": sha256_file(config_path),
            "dog_metas": portable_input_path(metas_path),
            "dog_metas_sha256": sha256_file(metas_path),
            "candidate_subset_sha256": candidate_subset_hash,
            "metadata_rows": metadata_rows,
            "unique_notices": len(joined),
            "fixed_candidate_count": len(candidate_ids),
            "fixed_candidate_ids": candidate_ids,
        },
        "summary": {
            "passed": not contract_failure_details,
            "failure_count": len(contract_failure_details),
            "failure_details": contract_failure_details,
            "profile_count": len(profile_reports),
            "profiles_with_different_top1_from_baseline": sum(
                profile["rank_change"]["top1_changed"] for profile in profile_reports
            ),
            "distinct_profile_top1_count": len(set(top1_values)),
            "mean_precision_at_5_delta": round(
                sum(precision_deltas) / len(precision_deltas), 6
            ),
            "mean_ndcg_at_5_delta": round(sum(ndcg_deltas) / len(ndcg_deltas), 6),
            "mean_mrr_delta": round(sum(mrr_deltas) / len(mrr_deltas), 6),
            "unknown_neutrality_failures": len(neutrality_failures),
            "unknown_neutrality_failed_profiles": neutrality_failures,
            "evidence_consistency_failures": len(all_consistency_failures),
            "evidence_consistency_failure_details": all_consistency_failures,
            "silver_score_mismatches": score_mismatch_count,
            "adoption_suitability_certainty_phrase_hits": len(all_certainty_hits),
            "adoption_suitability_certainty_hit_details": all_certainty_hits,
            "quality_sensitivity_failures": len(quality_sensitivity_failures),
        },
        "profiles": profile_reports,
        "quality_weight_sensitivity": quality_sensitivity,
        "limitations": [
            "This is a deterministic implementation contract, not independent evidence that reranking improves user outcomes or search relevance.",
            "Silver labels are deterministic proxies derived from public notice fields, not human judgments of adoption suitability.",
            "The fixed candidate pool isolates reranking behavior; it does not evaluate retrieval recall or population-level generalization.",
            "The quality-weight appendix compares ordering sensitivity on 12 fixed candidates and two profiles; it is not evidence of performance improvement or an optimal weight.",
            "Photo quality is a technical image-usability heuristic, not animal quality or adoption suitability.",
            "Only size, age group, and region are scored. Personality, activity, absence time, children, and other-pet compatibility are not inferred or scored.",
            "Notice state and metadata reflect the fixed reference date; shelter confirmation is still required before a visit.",
        ],
    }


def _metric(value: float) -> str:
    return f"{value:.3f}"


def _optional_metric(value: Any, digits: int = 6) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _candidate_summary(candidate: Mapping[str, Any]) -> str:
    appearance = candidate["silver_label"]["appearance"]
    return (
        f"{candidate['notice_id']} "
        f"({appearance['size']}/{appearance['age_group']}/{appearance['region']})"
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    sensitivity = report["quality_weight_sensitivity"]
    lines = [
        "# 외형 프로필 재정렬 결정적 계약 검증 — appearance_v1",
        "",
        (
            "> 이 보고서는 구현 규칙이 의도대로 동작하는지 확인하는 계약 검증이며, "
            "독립적인 검색 성능·사용자 효용 평가가 아닙니다."
        ),
        "",
        f"- 계약 결과: **{'PASS' if summary['passed'] else 'FAIL'}**",
        f"- 계약 실패: **{summary['failure_count']}건**",
        f"- 평가 기준일: `{report['evaluation_reference_date']}`",
        f"- 고정 후보: {report['inputs']['fixed_candidate_count']}개 공고",
        f"- 프로필: {summary['profile_count']}개",
        "- 실제 호출 함수: `app.profile_rerank.rerank_candidates`",
        "- 점수 범위: 크기·연령·지역만 사용하며 생활환경·성격 정보는 점수화하지 않음",
        "",
        "## 진단 지표 요약",
        "",
        "| 프로필 | P@5 기준→재정렬 | nDCG@5 기준→재정렬 | MRR 기준→재정렬 | 평균 순위 변동 |",
        "|---|---:|---:|---:|---:|",
    ]
    for profile in report["profiles"]:
        baseline = profile["baseline_metrics"]
        reranked = profile["reranked_metrics"]
        lines.append(
            "| {label} | {bp}→{rp} | {bn}→{rn} | {bm}→{rm} | {shift:.2f} |".format(
                label=profile["label"],
                bp=_metric(baseline["precision_at_5"]),
                rp=_metric(reranked["precision_at_5"]),
                bn=_metric(baseline["ndcg_at_5"]),
                rn=_metric(reranked["ndcg_at_5"]),
                bm=_metric(baseline["mrr"]),
                rm=_metric(reranked["mrr"]),
                shift=profile["rank_change"]["mean_absolute_rank_change"],
            )
        )

    lines.extend(
        [
            "",
            f"프로필별 재정렬 1위는 {summary['distinct_profile_top1_count']}개로 서로 달랐고, "
            f"기준 순서와 1위가 바뀐 프로필은 {summary['profiles_with_different_top1_from_baseline']}개입니다.",
            "",
            "## 동일 후보군의 프로필 A/B 순서 변화",
            "",
            "아래 두 열은 완전히 같은 후보와 검색 점수를 사용한 결과입니다.",
            "",
        ]
    )
    first, second = report["profiles"][:2]
    first_by_id = {row["notice_id"]: row for row in first["candidates"]}
    second_by_id = {row["notice_id"]: row for row in second["candidates"]}
    lines.extend(
        [
            f"| 순위 | {first['label']} | {second['label']} |",
            "|---:|---|---|",
        ]
    )
    for index in range(
        min(5, len(first["reranked_order"]), len(second["reranked_order"]))
    ):
        first_id = first["reranked_order"][index]
        second_id = second["reranked_order"][index]
        lines.append(
            f"| {index + 1} | {_candidate_summary(first_by_id[first_id])} | "
            f"{_candidate_summary(second_by_id[second_id])} |"
        )

    sensitivity_settings = sensitivity["settings"]
    sensitivity_summary = sensitivity["summary"]
    coverage = sensitivity["quality_coverage"]
    control_weight = sensitivity_settings["control_quality_weight"]
    production_weight = sensitivity_settings["production_default_quality_weight"]
    lines.extend(
        [
            "",
            "## 사진 품질 가중치 결정적 민감도 부록",
            "",
            (
                "> 같은 12개 후보와 두 프로필에서 품질 가중치만 바꾼 순서 민감도 계약입니다. "
                "검색 성능 향상, 사용자 효용 또는 최적 가중치를 주장하는 평가가 아닙니다."
            ),
            "",
            f"- 호환성 격리 계약: `quality_weight={control_weight:.2f}`",
            f"- 코드의 생산 기본값 비교: `quality_weight={production_weight:.2f}`",
            f"- 민감도 계약 결과: **{'PASS' if sensitivity_summary['passed'] else 'FAIL'}**",
            (
                f"- 사진 품질 known/unknown: **{coverage['known_count']}/"
                f"{coverage['unknown_count']}개** "
                f"({coverage['known_rate'] * 100:.2f}%/"
                f"{coverage['unknown_rate'] * 100:.2f}%)"
            ),
            "",
            (
                "| 프로필 | Top1 0.00→0.05 | Top-5 진입·이탈 (교체) | "
                "Top-5 내 순서 변경 후보 | 전체 평균 절대 이동 | 전체 최대 이동 |"
            ),
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for profile in sensitivity["profiles"]:
        top1 = profile["top1"]
        membership = profile["top_k_membership"]
        position = profile["top_k_position"]
        change = profile["all_candidate_rank_change"]
        lines.append(
            "| {label} | {control}→{production} ({changed}) | "
            "진입 {entered}개·이탈 {exited}개 ({replacements}쌍) | "
            "{positions}개 | "
            "{mean:.2f}위 | {maximum}위 |".format(
                label=profile["label"],
                control=top1["control_notice_id"],
                production=top1["production_default_notice_id"],
                changed="변화" if top1["changed"] else "불변",
                entered=len(membership["entered_notice_ids"]),
                exited=len(membership["exited_notice_ids"]),
                replacements=membership["replacement_count"],
                positions=position["changed_candidate_count"],
                mean=change["mean_absolute_rank_change"],
                maximum=change["max_absolute_rank_change"],
            )
        )

    lines.extend(
        [
            "",
            "### Top-5 순서 상세",
            "",
            f"| 프로필 | 품질 {control_weight:.2f} | 품질 {production_weight:.2f} |",
            "|---|---|---|",
        ]
    )
    for profile in sensitivity["profiles"]:
        membership = profile["top_k_membership"]
        lines.append(
            f"| {profile['label']} | "
            f"{' → '.join(membership['control_notice_ids'])} | "
            f"{' → '.join(membership['production_default_notice_ids'])} |"
        )

    unknown_rows = [
        (profile, row)
        for profile in sensitivity["profiles"]
        for row in profile["unknown_quality_candidates"]
    ]
    lines.extend(
        [
            "",
            "### 사진 품질 unknown 후보의 상대 효과",
            "",
            (
                "unknown 후보에는 품질 가산점을 직접 더하지 않으므로 점수 변화는 0이어야 합니다. "
                "다만 known 후보는 가산점을 받을 수 있어, 직접 감점이 없어도 이웃 후보와의 "
                "점수 간격은 달라질 수 있습니다."
            ),
            "",
        ]
    )
    if unknown_rows:
        lines.extend(
            [
                (
                    "| 프로필 | 공고 | 순위 0.00→0.05 | 직접 점수 변화 | "
                    "상위 후보와 격차 | 하위 후보보다 앞선 격차 |"
                ),
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for profile, row in unknown_rows:
            control = row["control"]
            production = row["production_default"]
            lines.append(
                "| {label} | {notice_id} | {control_rank}→{production_rank} | "
                "{score_delta:+.6f} | {higher_control}→{higher_production} | "
                "{lower_control}→{lower_production} |".format(
                    label=profile["label"],
                    notice_id=row["notice_id"],
                    control_rank=control["rank"],
                    production_rank=production["rank"],
                    score_delta=row["score_delta"],
                    higher_control=_optional_metric(control["margin_to_higher"]),
                    higher_production=_optional_metric(production["margin_to_higher"]),
                    lower_control=_optional_metric(control["margin_over_lower"]),
                    lower_production=_optional_metric(production["margin_over_lower"]),
                )
            )
    else:
        lines.append("- 고정 후보군에는 사진 품질 unknown 후보가 없습니다.")

    lines.extend(
        [
            "",
            "## 안전성·근거 검증",
            "",
            f"- unknown 중립성 실패: **{summary['unknown_neutrality_failures']}건** "
            "(모든 외형 정보가 unknown인 통제 후보의 최종 점수 변화가 0인지 확인)",
            f"- 추천 근거 일관성 실패: **{summary['evidence_consistency_failures']}건**",
            f"- 실버 라벨과 구현 점수 불일치: **{summary['silver_score_mismatches']}건**",
            f"- 입양 적합도를 확정하는 문구 탐지: **{summary['adoption_suitability_certainty_phrase_hits']}건**",
            f"- 사진 품질 민감도 계약 실패: **{summary['quality_sensitivity_failures']}건**",
            "",
            "## 재현 정보",
            "",
            f"- 프로필 설정 SHA-256: `{report['inputs']['profile_config_sha256']}`",
            f"- `dog_metas.json` SHA-256: `{report['inputs']['dog_metas_sha256']}`",
            f"- 고정 후보 메타데이터 SHA-256: `{report['inputs']['candidate_subset_sha256']}`",
            "- 재실행: `python scripts/evaluate_profile_reranking.py`",
            "",
            "## 해석 한계",
            "",
            "이 검증은 공고 원문의 체중·나이·지역으로 만든 **실버 라벨**과 동일한 규칙을 확인하는 고정 후보 계약 시험입니다. "
            "따라서 P@5·nDCG·MRR 변화는 독립적인 성능 향상 근거가 아닙니다. "
            "사람이 판정한 입양 적합도 평가가 아니며, 전체 검색의 재현율이나 실제 입양 성과를 증명하지 않습니다. "
            "품질 민감도 부록도 12개 고정 후보·두 프로필과 합성된 고정 검색 점수만 사용하므로 성능 향상이나 최적 가중치를 증명하지 않습니다. "
            "사진 품질은 선명도·대비·노출·해상도 기반의 기술적 사용성 지표이지 개체의 품질이나 입양 적합도가 아닙니다. "
            "품질 unknown 후보는 한 개뿐이므로 상대 순위 효과를 일반화할 수 없습니다. "
            "성격, 활동량, 부재시간, 아동·다른 동물과의 생활 가능성은 사진이나 품종으로 추정하지 않았고 점수에도 넣지 않았습니다. "
            "공고 상태와 실제 생활 적합성은 방문 전 원문 및 보호소를 통해 다시 확인해야 합니다.",
            "",
        ]
    )
    return "\n".join(lines)


def serialize_report(report: Mapping[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2) + "\n"


def write_report(
    report: Mapping[str, Any], json_path: Path, markdown_path: Path
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(serialize_report(report), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
