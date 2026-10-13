"""Per-query relevance filter over retrieved top-k.

For each sub-query and its retrieved docs, an LLM is asked to select the
indices of truly-relevant entries. Failed parses fall back to keeping the
first few docs by retrieval order (degraded but non-empty).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterable

from ..llm import LLMClient
from ..standard_client import SearchResult
from . import prompts
from .decompose import _parse_json

log = logging.getLogger(__name__)


_MAX_DOCS_TO_LLM = 20
_FALLBACK_KEEP = 5


def _kept_indices(value: Any, limit: int) -> list[int]:
    """Normalise the model's `kept_results` into ascending, de-duplicated,
    in-range document indices.

    Ascending order restores *retrieval* order. Left as the model wrote them,
    duplicates would emit the same document twice — double-counting its
    contribution to the merged ranking (run.py's `1/log2(rank+2)` decay) and
    pushing every later document one position further down.

    Integral floats (`0.0`) are accepted; a non-empty field that yields nothing
    is logged rather than silently discarded.
    """
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    out: set[int] = set()
    for item in items:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            idx = item
        elif isinstance(item, float) and item.is_integer():
            idx = int(item)
        elif isinstance(item, str) and item.strip().lstrip("-").isdigit():
            idx = int(item.strip())
        else:
            continue
        if 0 <= idx < limit:
            out.add(idx)
    if items and not out:
        log.warning("filter: dropped unusable kept_results %r (limit %d)", value, limit)
    return sorted(out)


def _pretty_inline(r: SearchResult) -> str:
    """Lean4-style inline formatting that matches what the paper's filter
    LLM was prompted on."""
    name = ".".join(str(p) for p in r.result.name)
    sig = (r.result.signature or "").strip()
    value = (r.result.value or "").strip()
    kind = r.result.kind
    if kind == "definition":
        formal = f"def {name}{sig} {value}".rstrip()
    elif kind in ("structure", "class", "inductive", "abbrev", "axiom", "constructor", "recursor"):
        formal = f"{kind} {name}{sig} {value}".rstrip()
    else:
        formal = f"{kind} {name}{sig}".rstrip()
    informal = (r.result.informal_description or "").strip()
    informal_prefix = f"/-- {informal} -/" if informal else ""
    return f"```lean4\n{informal_prefix}\n{formal}\n```"


async def _filter_single(
    llm: LLMClient,
    *,
    formal_statement: str,
    step_description: str,
    query: str,
    docs: list[SearchResult],
) -> tuple[list[int], str]:
    """Return (kept_indices, reasoning) for one (query, docs) pair."""
    if not docs:
        return [], "no docs to filter"
    truncated = docs[:_MAX_DOCS_TO_LLM]
    results_str = "\n\n".join(f"[{i}] {_pretty_inline(d)}" for i, d in enumerate(truncated))
    prompt = prompts.FILTER.format(
        formal_statement=formal_statement,
        step_description=step_description,
        query=query,
        search_results=results_str,
    )
    try:
        response = await llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
        data = _parse_json(response)
        kept = _kept_indices(data.get("kept_results"), len(truncated))
        return kept, str(data.get("reasoning") or "")
    except Exception as e:
        return list(range(min(_FALLBACK_KEEP, len(truncated)))), f"filter failed ({type(e).__name__}: {e}); keeping top {_FALLBACK_KEEP}"


async def filter_results(
    llm: LLMClient,
    *,
    formal_statement: str,
    query_to_docs: dict[str, list[SearchResult]],
    query_to_step_desc: dict[str, str],
) -> tuple[dict[str, list[SearchResult]], str]:
    """Filter `query_to_docs` query-by-query in parallel.

    Returns (filtered_query_to_docs, combined_filter_reasoning).
    """
    if not query_to_docs:
        return {}, ""
    queries = list(query_to_docs.keys())
    tasks = [
        _filter_single(
            llm,
            formal_statement=formal_statement,
            step_description=query_to_step_desc.get(q, "(no step context)"),
            query=q,
            docs=query_to_docs[q],
        )
        for q in queries
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    filtered: dict[str, list[SearchResult]] = {}
    reasoning_lines: list[str] = []
    for q, (kept_indices, reasoning) in zip(queries, results):
        docs = query_to_docs[q]
        filtered[q] = [docs[i] for i in kept_indices if i < len(docs)]
        reasoning_lines.append(f"Query '{q}': {reasoning}")
    return filtered, "\n\n".join(reasoning_lines)


def union_unique_results(
    filtered: dict[str, list[SearchResult]],
    iter_in: Iterable[str] | None = None,
) -> list[tuple[str, SearchResult]]:
    """Return [(doc_id, SearchResult)] in first-seen order, deduped by doc_id.

    `doc_id = module_name::name`. `iter_in` controls iteration order over
    queries; defaults to `filtered.keys()`.
    """
    seen: set[str] = set()
    out: list[tuple[str, SearchResult]] = []
    keys = list(iter_in) if iter_in is not None else list(filtered.keys())
    for q in keys:
        for d in filtered.get(q, []):
            doc_id = f"{'.'.join(d.result.module_name)}::{'.'.join(d.result.name)}"
            if doc_id in seen:
                continue
            seen.add(doc_id)
            out.append((doc_id, d))
    return out
