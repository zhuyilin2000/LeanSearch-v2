"""SiliconFlow API clients for the pg backend.

One module shared by the corpus loader (``corpus/load_pg.py``) and the serving
pipeline (``pg_pipeline.py``): synchronous httpx, exponential backoff on
429/5xx/network errors, and loud validation of response shapes.

Config lives under ``siliconflow.*`` in config.yaml / config.local.yaml:
the api key is read from the inline ``siliconflow.api_key`` value or, failing
that, from the env var named by ``siliconflow.api_key_env`` (default
``SILICONFLOW_API_KEY``) — the same indirection pattern ``llm.py`` uses for
LLM profiles.
"""

from __future__ import annotations

import math
import os
import random
import time
from typing import Any

import httpx

from .config import get, get_bool, get_int


class SiliconFlowError(RuntimeError):
    pass


def _resolve_api_key() -> str:
    inline = get("SILICONFLOW_API_KEY_INLINE", "siliconflow", "api_key", default=None)
    if inline not in (None, ""):
        return str(inline)
    key_env = str(get("SILICONFLOW_API_KEY_ENV", "siliconflow", "api_key_env", default="SILICONFLOW_API_KEY"))
    api_key = os.environ.get(key_env)
    if not api_key:
        raise SiliconFlowError(
            f"SiliconFlow api key not set: env var {key_env} is missing "
            f"(or set `siliconflow.api_key` in config.local.yaml)"
        )
    return api_key


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(math.fsum(x * x for x in vec))
    if norm == 0.0:
        raise SiliconFlowError("embedding has zero norm")
    return [x / norm for x in vec]


class _RetryCapableClient:
    """httpx.Client wrapper with backoff on 429 / 5xx / network errors."""

    RETRYABLE_STATUS = {429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str,
        timeout_s: float,
        max_retries: int,
        backoff_base_s: float = 1.0,
        backoff_cap_s: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
        )

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.post(path, json=payload)
                if resp.status_code in self.RETRYABLE_STATUS:
                    retry_after = resp.headers.get("Retry-After")
                    wait = self._backoff_delay(attempt)
                    if retry_after is not None:
                        try:
                            wait = max(wait, float(retry_after))
                        except ValueError:
                            pass
                    if attempt < self.max_retries:
                        time.sleep(wait)
                        continue
                    raise SiliconFlowError(
                        f"{path}: HTTP {resp.status_code} after {self.max_retries + 1} attempts: {resp.text[:300]}"
                    )
                if resp.status_code >= 400:
                    # Non-retryable client error: fail immediately with details.
                    raise SiliconFlowError(f"{path}: HTTP {resp.status_code}: {resp.text[:300]}")
                try:
                    data = resp.json()
                except ValueError as e:
                    # A <400 status with a non-JSON body (gateway/WAF HTML page,
                    # truncated response, empty 200) must surface as
                    # SiliconFlowError so callers can apply their fallback.
                    # Letting JSONDecodeError escape bypassed every
                    # `except SiliconFlowError` handler in the serving path.
                    raise SiliconFlowError(
                        f"{path}: HTTP {resp.status_code} but the body was not JSON "
                        f"({type(e).__name__}): {resp.text[:300]!r}"
                    ) from e
                if not isinstance(data, dict):
                    raise SiliconFlowError(
                        f"{path}: expected a JSON object, got {type(data).__name__}: {str(data)[:200]}"
                    )
                return data
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self._backoff_delay(attempt))
                    continue
                raise SiliconFlowError(f"{path}: network error after {self.max_retries + 1} attempts: {e}") from e
        raise SiliconFlowError(f"{path}: exhausted retries ({last_err})")

    def _backoff_delay(self, attempt: int) -> float:
        delay = min(self.backoff_base_s * (2**attempt), self.backoff_cap_s)
        return delay * (0.5 + 0.5 * random.random())

    def close(self) -> None:
        self._client.close()


class SiliconFlowEmbedder:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.siliconflow.cn/v1",
        model: str = "Qwen/Qwen3-Embedding-8B",
        dimensions: int = 1024,
        timeout_s: float = 120.0,
        max_retries: int = 6,
        backoff_base_s: float = 1.0,
        backoff_cap_s: float = 60.0,
    ) -> None:
        self.model = model
        self.dimensions = int(dimensions)
        self.total_prompt_tokens = 0
        self.total_requests = 0
        self._http = _RetryCapableClient(
            api_key,
            base_url=base_url,
            timeout_s=timeout_s,
            max_retries=max_retries,
            backoff_base_s=backoff_base_s,
            backoff_cap_s=backoff_cap_s,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts; returns L2-normalized vectors in input order."""
        if not texts:
            return []
        payload: dict[str, Any] = {
            "model": self.model,
            "input": list(texts),
            "dimensions": self.dimensions,
            "encoding_format": "float",
        }
        data = self._http.post("/embeddings", payload)
        items = data.get("data")
        if not isinstance(items, list) or len(items) != len(texts):
            raise SiliconFlowError(
                f"/embeddings: expected {len(texts)} items, got {len(items) if isinstance(items, list) else type(items)}"
            )
        out: list[list[float] | None] = [None] * len(texts)
        for item in items:
            idx = item.get("index")
            emb = item.get("embedding")
            if not isinstance(idx, int) or not 0 <= idx < len(texts):
                raise SiliconFlowError(f"/embeddings: bad index {idx!r} in response")
            if not isinstance(emb, list) or len(emb) != self.dimensions:
                raise SiliconFlowError(
                    f"/embeddings: expected {self.dimensions}-dim vector, got "
                    f"{len(emb) if isinstance(emb, list) else type(emb)} — was `dimensions` ignored?"
                )
            out[idx] = _normalize(emb)
        if any(v is None for v in out):
            raise SiliconFlowError("/embeddings: response missing some indices")
        usage = data.get("usage") or {}
        self.total_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.total_requests += 1
        return out  # type: ignore[return-value]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def close(self) -> None:
        self._http.close()


class SiliconFlowReranker:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.siliconflow.cn/v1",
        model: str = "Qwen/Qwen3-Reranker-8B",
        timeout_s: float = 120.0,
        max_retries: int = 1,
        backoff_base_s: float = 1.0,
        backoff_cap_s: float = 60.0,
        max_documents_per_request: int | None = None,
        use_instruction: bool = True,
        instruction: str | None = None,
    ) -> None:
        self.model = model
        self.max_documents_per_request = max_documents_per_request
        self.use_instruction = use_instruction
        self.instruction = instruction
        self._http = _RetryCapableClient(
            api_key,
            base_url=base_url,
            timeout_s=timeout_s,
            max_retries=max_retries,
            backoff_base_s=backoff_base_s,
            backoff_cap_s=backoff_cap_s,
        )

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
        instruction: str | None = None,
    ) -> list[tuple[int, float]]:
        """Score (query, doc) pairs.

        Returns ``[(original_document_index, score)]`` sorted by score desc
        (stable for ties). Documents are chunked by ``max_documents_per_request``
        if set; scores from the same model and query are comparable across
        chunk calls, so the merge is a plain sort.
        """
        if not documents:
            return []
        # Config beats the caller's argument here. `pg_pipeline` passes the
        # standard INSTRUCTION as a *default*, so letting the explicit argument
        # win made both `use_instruction` and `instruction` dead knobs on the
        # serving path: setting `use_instruction: false` still sent the
        # instruction, and a custom `instruction` was silently ignored.
        if not self.use_instruction:
            effective_instruction: str | None = None
        elif self.instruction is not None:
            effective_instruction = self.instruction
        else:
            effective_instruction = instruction
        chunk = self.max_documents_per_request or len(documents)
        merged: list[tuple[int, float]] = []
        for start in range(0, len(documents), chunk):
            docs = documents[start : start + chunk]
            payload: dict[str, Any] = {
                "model": self.model,
                "query": query,
                "documents": docs,
                "top_n": top_n if top_n is not None else len(docs),
                "return_documents": False,
            }
            if effective_instruction:
                payload["instruction"] = effective_instruction
            data = self._http.post("/rerank", payload)
            results = data.get("results")
            if not isinstance(results, list):
                raise SiliconFlowError(f"/rerank: no `results` list in response: {list(data)[:10]}")
            for item in results:
                idx = item.get("index")
                score = item.get("relevance_score", item.get("score"))
                if not isinstance(idx, int) or not 0 <= idx < len(docs):
                    raise SiliconFlowError(f"/rerank: bad index {idx!r} in response item {item}")
                if not isinstance(score, (int, float)):
                    raise SiliconFlowError(
                        f"/rerank: no relevance_score in response item (keys: {list(item)})"
                    )
                merged.append((start + idx, float(score)))
        merged.sort(key=lambda t: -t[1])
        return merged

    def close(self) -> None:
        self._http.close()


def embedder_from_config() -> SiliconFlowEmbedder:
    return SiliconFlowEmbedder(
        _resolve_api_key(),
        base_url=str(get("SILICONFLOW_BASE_URL", "siliconflow", "base_url", default="https://api.siliconflow.cn/v1")),
        model=str(get("SILICONFLOW_EMBEDDING_MODEL", "siliconflow", "embedding", "model", default="Qwen/Qwen3-Embedding-8B")),
        dimensions=int(get("SILICONFLOW_EMBEDDING_DIMENSIONS", "siliconflow", "embedding", "dimensions", default=1024)),
        timeout_s=float(get("SILICONFLOW_EMBEDDING_TIMEOUT_S", "siliconflow", "embedding", "timeout_s", default=120)),
        max_retries=int(get("SILICONFLOW_EMBEDDING_MAX_RETRIES", "siliconflow", "embedding", "max_retries", default=6)),
        backoff_base_s=float(get("SILICONFLOW_EMBEDDING_BACKOFF_BASE_S", "siliconflow", "embedding", "backoff_base_s", default=1.0)),
        backoff_cap_s=float(get("SILICONFLOW_EMBEDDING_BACKOFF_CAP_S", "siliconflow", "embedding", "backoff_cap_s", default=60.0)),
    )


def reranker_from_config() -> SiliconFlowReranker:
    return SiliconFlowReranker(
        _resolve_api_key(),
        base_url=str(get("SILICONFLOW_BASE_URL", "siliconflow", "base_url", default="https://api.siliconflow.cn/v1")),
        model=str(get("SILICONFLOW_RERANK_MODEL", "siliconflow", "rerank", "model", default="Qwen/Qwen3-Reranker-8B")),
        timeout_s=float(get("SILICONFLOW_RERANK_TIMEOUT_S", "siliconflow", "rerank", "timeout_s", default=120)),
        max_retries=int(get("SILICONFLOW_RERANK_MAX_RETRIES", "siliconflow", "rerank", "max_retries", default=1)),
        backoff_base_s=float(get("SILICONFLOW_RERANK_BACKOFF_BASE_S", "siliconflow", "rerank", "backoff_base_s", default=1.0)),
        backoff_cap_s=float(get("SILICONFLOW_RERANK_BACKOFF_CAP_S", "siliconflow", "rerank", "backoff_cap_s", default=60.0)),
        max_documents_per_request=get_int(
            # Canonical name follows the section/subsection/key convention. The
            # short legacy name is still honoured because it was the only one
            # that ever took effect (it crashed, which is now fixed).
            "SILICONFLOW_RERANK_MAX_DOCUMENTS_PER_REQUEST",
            "siliconflow", "rerank", "max_documents_per_request", default=None,
        ) or get_int(
            "SILICONFLOW_RERANK_MAX_DOCS",
            "siliconflow", "rerank", "max_documents_per_request", default=None,
        ),
        use_instruction=get_bool(
            "SILICONFLOW_RERANK_USE_INSTRUCTION", "siliconflow", "rerank", "use_instruction", default=True
        ),
        instruction=get("SILICONFLOW_RERANK_INSTRUCTION", "siliconflow", "rerank", "instruction", default=None),
    )
