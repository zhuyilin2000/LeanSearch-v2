"""Decompose a theorem into search sub-queries.

Two modes:
- Initial: theorem -> definitions / high-level / step-wise queries.
- Reflect: same task conditioned on a previously-rejected plan and the
  judge's feedback (used inside the big-loop reflection cycle).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..llm import LLMClient
from . import prompts


def _string_list(value: Any) -> list[str]:
    """Coerce a JSON field into a list of non-empty query strings.

    Models occasionally answer with a bare string where an array was requested.
    ``list("The definition of a UFD")`` would silently explode into 21
    one-character queries, each costing a retrieval plus a filter LLM call.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, (list, tuple)):
        return []
    return [s.strip() for s in value if isinstance(s, str) and s.strip()]


def _step_list(value: Any) -> list[dict[str, Any]]:
    """Keep only the dict-shaped entries of a ``steps`` field."""
    if not isinstance(value, (list, tuple)):
        return []
    return [s for s in value if isinstance(s, dict)]


@dataclass
class ProofPlan:
    definition_queries: list[str] = field(default_factory=list)
    highlevel_queries: list[str] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""

    @property
    def step_queries(self) -> list[str]:
        out: list[str] = []
        for s in self.steps:
            out.extend(_string_list(s.get("queries")))
        return out

    @property
    def all_queries(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for q in self.definition_queries + self.highlevel_queries + self.step_queries:
            if q not in seen:
                seen.add(q)
                out.append(q)
        return out


def _parse_json(response: str) -> dict[str, Any]:
    """Extract the first JSON *object* from `response`. Accepts plain JSON or
    fenced ```json blocks. Raises on parse failure (caller retries upstream).

    Every candidate is checked with `isinstance(obj, dict)`: a response whose
    fenced block holds a bare array/scalar is valid JSON, and returning it would
    hand a list to callers that immediately call `.get()`.
    """
    text = response or ""
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = fenced + [text]
    for cand in candidates:
        try:
            obj = json.loads(cand.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    raise json.JSONDecodeError("no JSON object found in response", text, 0)


async def decompose(
    llm: LLMClient,
    *,
    formal_statement: str,
    informal_statement: str,
    informal_proof: str = "",
    previous_plan: ProofPlan | None = None,
    previous_quality_reasoning: str = "",
) -> ProofPlan:
    if previous_plan is None:
        prompt = prompts.DECOMPOSE_INITIAL.format(
            formal_statement=formal_statement,
            informal_statement=informal_statement,
            informal_proof=informal_proof,
        )
    else:
        previous_steps_str = "\n".join(
            f"Step {i}: {s.get('description', '')}\n  Reasoning: {s.get('reasoning', '')}"
            for i, s in enumerate(previous_plan.steps)
        )
        prompt = prompts.DECOMPOSE_REFLECT.format(
            formal_statement=formal_statement,
            informal_statement=informal_statement,
            informal_proof=informal_proof,
            previous_steps=previous_steps_str,
            quality_reasoning=previous_quality_reasoning,
        )
    response = await llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    data = _parse_json(response)
    return ProofPlan(
        definition_queries=_string_list(data.get("definition_queries")),
        highlevel_queries=_string_list(data.get("highlevel_queries")),
        steps=_step_list(data.get("steps")),
        raw_response=response,
    )
