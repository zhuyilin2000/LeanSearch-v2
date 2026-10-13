"""CPU-only retrieval pipeline for the pg backend: SiliconFlow embeddings +
pgvector search + SiliconFlow rerank.

Mirrors ``pipeline.RetrievalPipeline``'s public surface exactly (``search()``
signature, ``SearchResult`` payloads, profile timing keys) so ``server.py``
and every downstream consumer (StandardClient, reasoning mode, prove task,
reproduce scripts) work unchanged. The only observable differences: the
``retrieval_backend`` profile value and ``/health``'s ``index_backend``.

Search order and semantics copied from ``pipeline.py``:

- query embedding text is ``f"Instruction: {INSTRUCTION}. Query: {query}"``,
  L2-normalized;
- ``retrieve_k`` defaults to ``max(top_k * 2, 50)`` when reranking, else
  ``top_k``;
- rerank documents are the stored ``text_to_embed`` values (the canonical
  ``construct_lean_representation`` output — the same texts the v2 reranker
  scored);
- ``distance`` in results is the *retrieval* distance of the candidate and
  is not updated after rerank reorders;
- on rerank failure the retriever order survives (rank-decreasing
  fallback scores), with a warning logged.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from psycopg_pool import ConnectionPool

from . import pgdb
from .config import get
from .siliconflow import (
    SiliconFlowEmbedder,
    SiliconFlowError,
    SiliconFlowReranker,
    embedder_from_config,
    reranker_from_config,
)
from .standard_client import INSTRUCTION, ResultData, SearchResult

log = logging.getLogger("leansearchv2.pg_pipeline")

RETRIEVE_SQL = """
SELECT id, src_key, name, module_name, kind, "type", signature, "value",
       informal_name, informal_description, text_to_embed,
       embedding <=> %s AS distance
FROM entries
ORDER BY embedding <=> %s
LIMIT %s
"""


class PGRetrievalPipeline:
    INSTRUCTION = INSTRUCTION
    index_backend = "pgvector"
    num_gpus = 0  # keeps /health's shape identical to the cuVS backend

    def __init__(
        self,
        *,
        pool: ConnectionPool,
        embedder: SiliconFlowEmbedder,
        reranker: SiliconFlowReranker,
        ef_search: int = 512,
        exact_scan: bool = False,
    ) -> None:
        self._pool = pool
        self.embedder = embedder
        self.reranker = reranker
        self.ef_search = int(ef_search)
        self.exact_scan = bool(exact_scan)

        with self._pool.connection() as conn:
            self.num_entries, self.num_embedded = conn.execute(
                "SELECT count(*), count(embedding) FROM entries"
            ).fetchone()
        self.num_entries, self.num_embedded = int(self.num_entries), int(self.num_embedded)
        print(
            f"pg backend: {self.num_embedded}/{self.num_entries} entries embedded"
            + (
                f" — WARNING: {self.num_entries - self.num_embedded} rows lack embeddings "
                "(retrieval quality degraded until `load_pg embed` finishes)"
                if self.num_embedded < self.num_entries else ""
            )
        )

    @classmethod
    def from_config(cls) -> "PGRetrievalPipeline":
        pool = pgdb.pool(
            min_size=int(get("PG_POOL_MIN", "pg", "pool_min", default=1)),
            max_size=int(get("PG_POOL_MAX", "pg", "pool_max", default=8)),
        )
        return cls(
            pool=pool,
            embedder=embedder_from_config(),
            reranker=reranker_from_config(),
            ef_search=int(get("PG_EF_SEARCH", "pg", "ef_search", default=512)),
            exact_scan=bool(get("PG_EXACT_SCAN", "pg", "exact_scan", default=False)),
        )

    def _configure_conn(self, conn) -> None:
        # hnsw.ef_search only matters once a HNSW index exists; harmless otherwise.
        conn.execute(f"SET hnsw.ef_search = {int(self.ef_search)}")

    def _retrieve(self, emb: list[float], k: int) -> list[dict[str, Any]]:
        """Top-k rows by cosine distance. Without a HNSW index this is an
        exact scan (recall 1.0); with one, pgvector picks the index unless
        ``exact_scan`` forces a seq scan."""
        from pgvector.psycopg import Vector

        vec = Vector(emb)  # plain list[float] would adapt to double precision[]
        with self._pool.connection() as conn:
            self._configure_conn(conn)
            if self.exact_scan:
                with conn.transaction():
                    conn.execute("SET LOCAL enable_indexscan = off")
                    cur = conn.execute(RETRIEVE_SQL, (vec, vec, k))
                    rows = cur.fetchall()
            else:
                rows = conn.execute(RETRIEVE_SQL, (vec, vec, k)).fetchall()
        cols = (
            "id", "src_key", "name", "module_name", "kind", "type", "signature",
            "value", "informal_name", "informal_description", "text_to_embed",
            "distance",
        )
        return [dict(zip(cols, r)) for r in rows]

    def _build_result(self, row: dict[str, Any]) -> SearchResult:
        return SearchResult(
            result=ResultData(
                module_name=row["module_name"],
                kind=row["kind"],
                name=row["name"],
                signature=row["signature"],
                type=row["type"],
                value=row["value"],
                docstring=None,  # the released corpus carries no docstring
                informal_name=row["informal_name"],
                informal_description=row["informal_description"],
            ),
            distance=float(row["distance"]),
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        return_profile: bool = False,
        rerank: bool = True,
        retrieve_k: int | None = None,
    ) -> list[SearchResult] | tuple[list[SearchResult], dict]:
        profile: dict = {}

        t0 = time.time()
        q = f"Instruction: {self.INSTRUCTION}. Query: {query}"
        emb = self.embedder.embed_one(q)
        profile["embedding_time"] = time.time() - t0

        if retrieve_k is None:
            retrieve_k = max(top_k * 2, 50) if rerank else top_k

        t0 = time.time()
        rows = self._retrieve(emb, retrieve_k)
        candidates = [(r, float(r["distance"])) for r in rows]
        profile["retrieval_time"] = time.time() - t0
        profile["retrieval_backend"] = "pgvector"
        profile["num_candidates"] = len(candidates)
        profile["reranked"] = bool(rerank)

        if not candidates:
            if return_profile:
                profile["rerank_time"] = 0.0
                profile["total_time"] = profile["embedding_time"] + profile["retrieval_time"]
                return [], profile
            return []

        if not rerank:
            t0 = time.time()
            results = [self._build_result(r) for r, _ in candidates[:top_k]]
            profile["postprocess_time"] = time.time() - t0
            profile["rerank_time"] = 0.0
            profile["total_time"] = (
                profile["embedding_time"] + profile["retrieval_time"] + profile["postprocess_time"]
            )
            if return_profile:
                return results, profile
            return results

        dist_map = {r["id"]: d for r, d in candidates}
        candidate_rows = [r for r, _ in candidates]
        n = len(candidate_rows)
        retriever_rank_scores = [(n - i) / n for i in range(n)]

        t0 = time.time()
        pairs = [r["text_to_embed"] for r in candidate_rows]
        profile["prompt_build_time"] = time.time() - t0

        t0 = time.time()
        scores = self._rerank(query, pairs, fallback_scores=retriever_rank_scores)
        profile["rerank_time"] = time.time() - t0

        t0 = time.time()
        ranked = sorted(zip(candidate_rows, scores), key=lambda x: -x[1])
        results = [self._build_result(r) for r, _ in ranked[:top_k]]
        profile["postprocess_time"] = time.time() - t0

        profile["total_time"] = (
            profile["embedding_time"]
            + profile["retrieval_time"]
            + profile["prompt_build_time"]
            + profile["rerank_time"]
            + profile["postprocess_time"]
        )

        if return_profile:
            return results, profile
        return results

    def _rerank(
        self,
        query: str,
        pairs: list[str],
        fallback_scores: list[float],
    ) -> list[float]:
        """Score every (query, doc) pair via the rerank API.

        Mirrors ``pipeline._rerank``'s contract: on failure the caller's
        fallback (rank-decreasing values in retriever order) is returned so
        the worst case degrades to retriever-only ranking. Documents the API
        omits from its response despite ``top_n=len(docs)`` are pushed just
        below every returned score, preserving retriever order among them.
        """
        if not pairs:
            return []
        try:
            ranked = self.reranker.rerank(
                query, pairs, top_n=len(pairs), instruction=self.INSTRUCTION
            )
        except SiliconFlowError as e:
            log.warning(f"rerank failed ({e}); falling back to retriever order")
            return list(fallback_scores)
        scores = list(fallback_scores)
        for orig_idx, score in ranked:
            scores[orig_idx] = score
        # Un-returned docs: fallback score minus 1 keeps them strictly below
        # any real API score (API scores are >= 0) while preserving order.
        returned = {i for i, _ in ranked}
        for i in range(len(pairs)):
            if i not in returned:
                scores[i] = fallback_scores[i] - 1.0
        return scores

    def close(self) -> None:
        try:
            self.embedder.close()
        finally:
            self.reranker.close()
            self._pool.close()
