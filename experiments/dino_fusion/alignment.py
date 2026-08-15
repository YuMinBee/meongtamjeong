"""Pure data helpers for CLIP-text-to-DINO alignment experiments."""

from __future__ import annotations

import hashlib
import importlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from app.graph_rag import infer_age_hint, infer_size_from_weight
from app.retrieval_evaluation import normalize_public_colors
from experiments.dino_fusion.core import clean_text, normalize_rows


SIZE_EN = {
    "tiny": "very small",
    "small": "small",
    "medium": "medium-sized",
    "large": "large",
}
SIZE_KO = {
    "tiny": "아주 작은",
    "small": "작은",
    "medium": "중간 크기",
    "large": "큰",
}
AGE_EN = {"puppy": "young", "adult": "adult", "senior": "senior"}
AGE_KO = {"puppy": "어린", "adult": "성견", "senior": "노령"}
COLOR_KO = {
    "black": "검정",
    "brown": "갈색",
    "cream": "크림색",
    "gold": "금색",
    "gray": "회색",
    "spotted": "반점",
    "tan": "황갈색",
    "white": "흰색",
    "yellow": "노란색",
}


def alignment_attributes(
    meta: Mapping[str, Any], *, reference_year: int = 2026
) -> dict[str, Any] | None:
    """Extract conservative appearance labels from public structured fields."""

    colors = normalize_public_colors(meta.get("color") or meta.get("colorCd"))
    size = clean_text(infer_size_from_weight(meta.get("weight")))
    age = clean_text(
        infer_age_hint(
            meta.get("age") or meta.get("ageCd"), reference_year=reference_year
        )
    )
    if not colors or not size:
        return None
    return {"colors": tuple(colors), "size": size, "age": age or "unknown"}


def semantic_signature(attributes: Mapping[str, Any]) -> str:
    colors = ",".join(str(value) for value in attributes.get("colors", ()))
    return f"colors={colors}|size={clean_text(attributes.get('size'))}|age={clean_text(attributes.get('age'))}"


def _words(attributes: Mapping[str, Any]) -> dict[str, str]:
    colors = tuple(str(value) for value in attributes.get("colors", ()))
    size = clean_text(attributes.get("size"))
    age = clean_text(attributes.get("age"))
    color_en = " and ".join(colors)
    color_ko = "과 ".join(COLOR_KO.get(value, value) for value in colors)
    return {
        "color_en": color_en,
        "color_ko": color_ko,
        "size_en": SIZE_EN.get(size, size),
        "size_ko": SIZE_KO.get(size, size),
        "age_en": AGE_EN.get(age, ""),
        "age_ko": AGE_KO.get(age, ""),
    }


def training_prompts(attributes: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Return bilingual training paraphrases for one semantic signature."""

    words = _words(attributes)
    age_en = f"{words['age_en']} " if words["age_en"] else ""
    age_ko = f"{words['age_ko']} " if words["age_ko"] else ""
    return {
        "english": (
            f"a photo of a {age_en}{words['size_en']} dog with {words['color_en']} fur",
            f"a {words['size_en']} {words['color_en']} {age_en}dog",
            f"a lost {age_en}dog with a {words['color_en']} coat and {words['size_en']} build",
        ),
        "korean": (
            f"{words['color_ko']} 털을 가진 {age_ko}{words['size_ko']} 강아지 사진",
            f"{words['size_ko']} {age_ko}강아지, 털 색은 {words['color_ko']}",
            f"{words['color_ko']} 색상의 {words['size_ko']} {age_ko}유실견",
        ),
    }


def evaluation_prompts(attributes: Mapping[str, Any]) -> dict[str, str]:
    """Return held-out phrasings that do not appear in ``training_prompts``."""

    words = _words(attributes)
    age_en = f" {words['age_en']}" if words["age_en"] else ""
    age_ko = f" {words['age_ko']}" if words["age_ko"] else ""
    return {
        "english": (
            f"find a {words['size_en']}{age_en} dog whose coat is {words['color_en']}"
        ),
        "korean": (
            f"{words['color_ko']} 털에 {words['size_ko']} 체격인{age_ko} 강아지를 찾아줘"
        ),
    }


def collect_heldout_notice_ids(report: Mapping[str, Any]) -> set[str]:
    """Collect the complete sampled pool, not only successful query downloads."""

    values: set[str] = set()
    for section in ("sample", "queries"):
        rows = report.get(section) or []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            dog_id = clean_text(row.get("notice_id") or row.get("desertionNo"))
            if dog_id:
                values.add(dog_id)
    return values


def stratified_notice_split(
    records: Sequence[Mapping[str, Any]],
    *,
    validation_fraction: float,
    seed: str,
) -> tuple[list[int], list[int]]:
    """Split within signatures, keeping singleton signatures in training."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        signature = clean_text(record.get("signature"))
        notice = clean_text(record.get("notice_id"))
        if not signature or not notice:
            raise ValueError("records require signature and notice_id")
        grouped[signature].append(index)

    train: list[int] = []
    validation: list[int] = []
    for signature, indices in sorted(grouped.items()):
        ordered = sorted(
            indices,
            key=lambda index: hashlib.sha256(
                f"{seed}|{signature}|{records[index]['notice_id']}".encode("utf-8")
            ).hexdigest(),
        )
        validation_count = (
            min(len(ordered) - 1, max(1, round(len(ordered) * validation_fraction)))
            if len(ordered) >= 2
            else 0
        )
        validation.extend(ordered[:validation_count])
        train.extend(ordered[validation_count:])
    return sorted(train), sorted(validation)


def make_projection_head(
    architecture: str,
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 512,
    flow_width: int = 96,
    flow_depth: int = 4,
    flow_steps: int = 8,
    flow_time_dim: int = 32,
) -> Any:
    torch = importlib.import_module("torch")
    nn = torch.nn
    if architecture == "linear":
        return nn.Linear(input_dim, output_dim, bias=False)
    if architecture == "mlp":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    if architecture == "flow":
        if min(flow_width, flow_depth, flow_steps, flow_time_dim) <= 0:
            raise ValueError("flow dimensions, depth, and steps must be positive")

        class TimeConditionedFlowHead(nn.Module):
            """Small hyperspherical ODE mapping CLIP text into DINO space.

            This is an independent retrieval-oriented implementation of the
            DINOde idea: an initial cross-encoder projection followed by a
            learned, time-conditioned tangent velocity field. The bottleneck
            width keeps its parameter budget close to the existing MLP.
            """

            def __init__(self) -> None:
                super().__init__()
                self.input_projection = nn.Linear(input_dim, output_dim, bias=False)
                self.time_mlp = nn.Sequential(
                    nn.Linear(flow_time_dim, flow_time_dim),
                    nn.GELU(),
                    nn.Linear(flow_time_dim, flow_time_dim),
                )
                self.velocity_input = nn.Linear(output_dim + flow_time_dim, flow_width)
                self.velocity_blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.LayerNorm(flow_width),
                            nn.Linear(flow_width, flow_width),
                            nn.GELU(),
                            nn.Linear(flow_width, flow_width),
                        )
                        for _ in range(flow_depth)
                    ]
                )
                self.velocity_output = nn.Linear(flow_width, output_dim)
                nn.init.zeros_(self.velocity_output.weight)
                nn.init.zeros_(self.velocity_output.bias)
                frequencies = torch.exp(
                    torch.linspace(
                        0.0,
                        math.log(1000.0),
                        max(1, flow_time_dim // 2),
                    )
                )
                self.register_buffer("time_frequencies", frequencies)
                self.steps = flow_steps
                self.dt = 1.0 / flow_steps
                self.flow_config = {
                    "width": flow_width,
                    "depth": flow_depth,
                    "steps": flow_steps,
                    "time_dim": flow_time_dim,
                    "integration": "euler_tangent_retraction",
                }

            def _time_embedding(self, t: Any, batch_size: int, values: Any) -> Any:
                if not torch.is_tensor(t):
                    t = torch.full(
                        (batch_size,),
                        float(t),
                        device=values.device,
                        dtype=values.dtype,
                    )
                else:
                    t = t.to(device=values.device, dtype=values.dtype)
                    if t.ndim == 0:
                        t = t.expand(batch_size)
                    elif t.shape != (batch_size,):
                        raise ValueError(
                            "flow time must be scalar or one value per row"
                        )
                angles = t[:, None] * self.time_frequencies[None, :].to(
                    dtype=values.dtype
                )
                embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
                if embedding.shape[1] < flow_time_dim:
                    embedding = torch.cat((embedding, t[:, None]), dim=-1)
                return self.time_mlp(embedding[:, :flow_time_dim])

            def initial_state(self, values: Any) -> Any:
                return torch.nn.functional.normalize(
                    self.input_projection(values), dim=-1
                )

            def velocity(self, state: Any, t: Any) -> Any:
                time_embedding = self._time_embedding(t, state.shape[0], state)
                hidden = torch.nn.functional.gelu(
                    self.velocity_input(torch.cat((state, time_embedding), dim=-1))
                )
                for block in self.velocity_blocks:
                    hidden = hidden + block(hidden)
                raw_velocity = self.velocity_output(hidden)
                radial = (raw_velocity * state).sum(dim=-1, keepdim=True)
                return raw_velocity - radial * state

            def forward(self, values: Any) -> Any:
                state = self.initial_state(values)
                for step in range(self.steps):
                    state = torch.nn.functional.normalize(
                        state + self.dt * self.velocity(state, step * self.dt),
                        dim=-1,
                    )
                return state

        return TimeConditionedFlowHead()
    raise ValueError("architecture must be 'linear', 'mlp', or 'flow'")


def project_embeddings(head: Any, values: Any) -> Any:
    torch = importlib.import_module("torch")
    return torch.nn.functional.normalize(head(values), dim=-1)


def projection_head_from_checkpoint(checkpoint: Mapping[str, Any]) -> Any:
    """Rebuild a projection head from its portable checkpoint metadata."""

    return make_projection_head(
        clean_text(checkpoint.get("architecture")),
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 512)),
        flow_width=int(checkpoint.get("flow_width", 96)),
        flow_depth=int(checkpoint.get("flow_depth", 4)),
        flow_steps=int(checkpoint.get("flow_steps", 8)),
        flow_time_dim=int(checkpoint.get("flow_time_dim", 32)),
    )


def blend_embeddings(image: Any, text: Any, *, text_weight: float) -> Any:
    """Blend comparable DINO-space vectors with a fixed convex weight."""

    if not 0.0 <= text_weight <= 1.0:
        raise ValueError("text_weight must be between zero and one")
    image_array = np.asarray(image, dtype=np.float32)
    text_array = np.asarray(text, dtype=np.float32)
    if image_array.shape != text_array.shape:
        raise ValueError("image and text embeddings must have the same shape")
    return normalize_rows((1.0 - text_weight) * image_array + text_weight * text_array)
