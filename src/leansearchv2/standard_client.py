"""HTTP client for the standard-mode retriever endpoint.

Reasoning mode and the prove task consume the retriever exclusively through
this client; nothing here imports `pipeline.py` (which pulls in cuVS/torch),
so the client is CPU-only and can run anywhere with network access.
"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel

from .config import get

# The retrieval instruction every query is prefixed with (both the cuVS and
# the pg pipeline embed `f"Instruction: {INSTRUCTION}. Query: {query}"`, and
# the rerank prompt carries it as its <Instruct> line). Lives here — the
# CPU-only contract module — so both pipelines share one source of truth.
INSTRUCTION = (
    "Retrieve the informal + formal representation of items in Mathlib4 "
    "that is mathematically relevant to the query. If the query ask for "
    "theorem or lemma, you shall try to find the entry that starts with "
    "theorem. If the query ask for a definition, you shall try to find "
    "the entry that starts with either definition or instance."
)


class ResultData(BaseModel):
    module_name: list[str]
    kind: str
    name: list[str]
    signature: str
    type: str
    value: str | None = None
    docstring: str | None = None
    informal_name: str | None = None
    informal_description: str | None = None


class SearchResult(BaseModel):
    result: ResultData
    distance: float


class StandardClient:
    def __init__(self, url: str | None = None, timeout: float | None = None) -> None:
        url = url or get("RETRIEVER_URL", "retriever", "url", default="http://localhost:8000")
        self.url = url.rstrip("/")
        self.timeout = float(
            timeout if timeout is not None else get("RETRIEVER_TIMEOUT_S", "retriever", "timeout_s", default=60)
        )

    async def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        rerank: bool = True,
        retrieve_k: int | None = None,
    ) -> list[SearchResult]:
        results = await self.search_batch([query], top_k, rerank=rerank, retrieve_k=retrieve_k)
        return results[0]

    async def search_batch(
        self,
        queries: list[str],
        top_k: int = 10,
        *,
        rerank: bool = True,
        retrieve_k: int | None = None,
    ) -> list[list[SearchResult]]:
        body: dict[str, Any] = {"query": queries, "num_results": top_k, "rerank": rerank}
        if retrieve_k is not None:
            body["retrieve_k"] = retrieve_k
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.url}/search", json=body)
            r.raise_for_status()
            data = r.json()
        return [[SearchResult.model_validate(item) for item in batch] for batch in data]

    async def health(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.get(f"{self.url}/health")
            r.raise_for_status()
            return r.json()
