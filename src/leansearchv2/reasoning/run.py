"""Reasoning-mode orchestrator.

Pure-asyncio implementation of the decompose → search → filter → judge
loop. Public entry point: `run_reasoning(problem, retriever, llm, ...)`.

Flow per problem:

    for attempt in range(big_loop + 1):
        plan = await decompose(...)           # LLM
        query_to_docs = await batch_search(...)   # HTTP (parallel)
        filtered = await filter_results(...)  # LLM (parallel)
        verdict = await judge(...)            # LLM
        if verdict == "good": status = "good"; break
        if attempt == big_loop: status = "fail"; break
        # else: keep going, feed plan+reasoning back into next decompose

The output is the union of `filtered` (dedup-by-doc-id) of the final
attempt, ranked by `aggregate_filtered_method1` (NDCG-discount summed
across sub-queries) so docs hit by multiple sub-queries surface first.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Any

from ..llm import LLMClient
from ..standard_client import SearchResult, StandardClient
from .decompose import ProofPlan, decompose
from .filter import filter_results, union_unique_results
from .judge import judge

log = logging.getLogger(__name__)


@dataclass
class ReasoningLLMs:
    """One LLMClient per reasoning-mode role; build via `from_config()` or
    pass clients directly to use the same model everywhere."""
    sketch: LLMClient
    filter: LLMClient
    judge: LLMClient

    @classmethod
    def from_config(cls) -> "ReasoningLLMs":
        from ..config import get
        sketch = str(get("REASONING_SKETCH_LLM", "reasoning", "sketch_llm", default="openai"))
        filt = str(get("REASONING_FILTER_LLM", "reasoning", "filter_llm", default="openai"))
        judge_ = str(get("REASONING_JUDGE_LLM", "reasoning", "judge_llm", default="openai"))
        cache: dict[str, LLMClient] = {}
        def _get(name: str) -> LLMClient:
            if name not in cache:
                cache[name] = LLMClient(name)
            return cache[name]
        return cls(sketch=_get(sketch), filter=_get(filt), judge=_get(judge_))


@dataclass
class Problem:
    problem_id: str
    formal_statement: str
    informal_statement: str = ""
    informal_proof: str = ""


@dataclass
class ReasoningResult:
    problem_id: str
    status: str  # "good" | "bad" | "fail"
    big_loop_count: int
    plan: ProofPlan
    filtered: dict[str, list[SearchResult]]
    filter_reasoning: str
    quality_reasoning: str
    entries: list[tuple[str, float, SearchResult]]
    """final ranked output: (doc_id, score, result) tuples, length <= output_top_k"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.problem_id,
            "sketch_status": self.status,
            "big_loop_count": self.big_loop_count,
            "definition_queries": self.plan.definition_queries,
            "highlevel_queries": self.plan.highlevel_queries,
            "informal_steps": [
                {
                    "step_id": i,
                    "description": s.get("description", ""),
                    "reasoning": s.get("reasoning", ""),
                    "queries": s.get("queries", []),
                }
                for i, s in enumerate(self.plan.steps)
            ],
            "filter_reasoning": self.filter_reasoning,
            "quality_reasoning": self.quality_reasoning,
            "entries": [
                {
                    "doc_id": did,
                    "score": score,
                    "name": ".".join(r.result.name),
                    "module": ".".join(r.result.module_name),
                    "kind": r.result.kind,
                    "informal_name": r.result.informal_name,
                    "distance": r.distance,
                }
                for did, score, r in self.entries
            ],
        }


def _step_desc_for_query(plan: ProofPlan, query: str) -> str:
    """Pick a human-readable step description to feed the filter prompt."""
    for q in plan.definition_queries:
        if q == query:
            return "Core definition used in the theorem statement"
    for q in plan.highlevel_queries:
        if q == query:
            return "High-level theorem that might directly bridge assumptions to conclusion"
    for i, s in enumerate(plan.steps):
        if query in (s.get("queries") or []):
            return f"Step {i}: {s.get('description', '')}"
    return "(no step context)"


async def _batch_search(
    retriever: StandardClient,
    queries: list[str],
    *,
    top_k: int,
    rerank: bool,
    concurrency: int,
) -> dict[str, list[SearchResult]]:
    """Fan out one /search per query (concurrency-bounded). Returns
    `{query: results}`. Empty-results queries map to `[]`."""
    if not queries:
        return {}
    sem = asyncio.Semaphore(concurrency)

    async def one(q: str) -> tuple[str, list[SearchResult]]:
        async with sem:
            try:
                return q, await retriever.search(q, top_k=top_k, rerank=rerank)
            except Exception as e:
                # A silent `[]` made a service failure indistinguishable from
                # "nothing relevant exists"; the run then scored 0 with no clue.
                log.warning(
                    "search failed for query %r (%s: %s); treating as no results",
                    q[:120], type(e).__name__, e,
                )
                return q, []

    pairs = await asyncio.gather(*[one(q) for q in queries])
    out: dict[str, list[SearchResult]] = {}
    for q, docs in pairs:
        out[q] = docs
    return out


def _rank_method1(
    filtered: dict[str, list[SearchResult]], top_k: int,
) -> list[tuple[str, float, SearchResult]]:
    """NDCG-discount summed across sub-queries. A doc that surfaces in
    multiple sub-queries (or near the top within one) accumulates score."""
    score: dict[str, float] = {}
    doc_ref: dict[str, SearchResult] = {}
    for _q, docs in filtered.items():
        for i, d in enumerate(docs):
            doc_id = f"{'.'.join(d.result.module_name)}::{'.'.join(d.result.name)}"
            score[doc_id] = score.get(doc_id, 0.0) + 1.0 / math.log2(i + 2)
            doc_ref.setdefault(doc_id, d)
    ranked = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [(did, sc, doc_ref[did]) for did, sc in ranked]


async def run_reasoning(
    problem: Problem,
    retriever: StandardClient,
    llms: ReasoningLLMs | LLMClient,
    *,
    search_top_k: int = 30,
    output_top_k: int = 100,
    big_loop: int = 3,
    retriever_concurrency: int = 8,
    retriever_rerank: bool = True,
) -> ReasoningResult:
    """Run the GetQuery -> Search -> Filter -> Judge loop until the judge says
    "good" or we exhaust the big loop budget. Pass a `LLMClient` to use one
    model for every role, or a `ReasoningLLMs` to split sketch/filter/judge.
    """
    if isinstance(llms, LLMClient):
        llms = ReasoningLLMs(sketch=llms, filter=llms, judge=llms)
    plan: ProofPlan | None = None
    prev_reasoning: str = ""
    filtered: dict[str, list[SearchResult]] = {}
    filter_reason: str = ""
    quality_reason: str = ""
    status: str = "fail"
    big_loop_count = 0

    for attempt in range(big_loop + 1):
        plan = await decompose(
            llms.sketch,
            formal_statement=problem.formal_statement,
            informal_statement=problem.informal_statement,
            informal_proof=problem.informal_proof,
            previous_plan=plan,
            previous_quality_reasoning=prev_reasoning,
        )
        all_queries = plan.all_queries
        query_to_docs = await _batch_search(
            retriever,
            all_queries,
            top_k=search_top_k,
            rerank=retriever_rerank,
            concurrency=retriever_concurrency,
        )
        query_to_step_desc = {q: _step_desc_for_query(plan, q) for q in all_queries}
        filtered, filter_reason = await filter_results(
            llms.filter,
            formal_statement=problem.formal_statement,
            query_to_docs=query_to_docs,
            query_to_step_desc=query_to_step_desc,
        )
        verdict, quality_reason = await judge(
            llms.judge,
            formal_statement=problem.formal_statement,
            plan_steps=plan.steps,
            filtered=filtered,
            filter_reasoning=filter_reason,
        )
        if verdict == "good":
            status = "good"
            break
        if attempt == big_loop:
            status = "fail"
            break
        big_loop_count += 1
        prev_reasoning = quality_reason

    entries = _rank_method1(filtered, top_k=output_top_k)
    return ReasoningResult(
        problem_id=problem.problem_id,
        status=status,
        big_loop_count=big_loop_count,
        plan=plan or ProofPlan(),
        filtered=filtered,
        filter_reasoning=filter_reason,
        quality_reasoning=quality_reason,
        entries=entries,
    )


# Re-export for convenience
__all__ = ["Problem", "ReasoningLLMs", "ReasoningResult", "run_reasoning", "union_unique_results"]
