"""FastAPI server for LeanSearch v2 standard mode.

Run with::

    ./scripts/serve.sh          # cuVS backend (CUDA machines)
    ./scripts/serve_pg.sh       # pg backend (SiliconFlow API + pgvector, CPU-only)

or directly with uvicorn::

    python -m uvicorn leansearchv2.server:app --host 0.0.0.0 --port 8000

Paths and tuning knobs are read from ``config.yaml`` at the repository root;
each can be overridden by an environment variable of the same name. The
backend is selected by ``serve.index_backend`` (config) / ``$INDEX_BACKEND``
env var: ``cuvs`` (default, needs the GPU stack) or ``pg``.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import dotenv
from fastapi import Body, FastAPI

from .config import get, get_path
from .standard_client import SearchResult  # same class pipeline.py re-exports


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    dotenv.load_dotenv()
    log.info("Starting...")
    backend = str(get("INDEX_BACKEND", "serve", "index_backend", default="cuvs")).lower()
    if backend in ("pg", "pgvector", "postgres"):
        from .pg_pipeline import PGRetrievalPipeline

        app.pipeline = PGRetrievalPipeline.from_config()
    else:
        from .pipeline import RetrievalPipeline

        num_gpus = get("NUM_GPUS", "serve", "num_gpus")
        app.pipeline = RetrievalPipeline(
            vectordb_dir=get_path("VECTORDB_DIR", "paths", "cuvs_db"),
            embedding_model_path=get_path("EMBEDDING_MODEL_PATH", "models", "embedder"),
            reranker_model_path=get_path("RERANKER_MODEL_PATH", "models", "reranker"),
            num_gpus=int(num_gpus) if num_gpus is not None else None,
            gpu_memory_utilization=float(
                get("GPU_MEMORY_UTILIZATION", "serve", "gpu_memory_utilization", default=0.9)
            ),
        )
    log.info("Ready (index_backend=%s)", app.pipeline.index_backend)
    try:
        yield
    finally:
        # Release the connection pool and the two httpx clients on shutdown;
        # `close()` had no caller at all, so they leaked until process exit.
        close = getattr(app.pipeline, "close", None)
        if close is not None:
            try:
                close()
            except Exception as e:  # shutdown must not raise
                log.warning("pipeline close failed: %s", e)
        app.pipeline = None


app = FastAPI(lifespan=lifespan)


@app.post("/search")
def search(
    query: list[str] = Body(...),
    num_results: int = Body(default=10, ge=1, le=1000),
    rerank: bool = Body(default=True),
    retrieve_k: int | None = Body(default=None, ge=1, le=10000),
) -> list[list[SearchResult]]:
    t0 = time.time()
    results = [
        app.pipeline.search(q, num_results, rerank=rerank, retrieve_k=retrieve_k)
        for q in query
    ]
    log.info(
        f"/search | {time.time() - t0:.2f}s | n={len(query)} | rerank={rerank} | {query}"
    )
    return results


@app.post("/search_with_profile")
def search_with_profile(
    query: list[str] = Body(...),
    num_results: int = Body(default=10, ge=1, le=1000),
    rerank: bool = Body(default=True),
    retrieve_k: int | None = Body(default=None, ge=1, le=10000),
) -> dict:
    t0 = time.time()
    all_results = []
    all_profiles = []
    for q in query:
        results, profile = app.pipeline.search(
            q, num_results, return_profile=True, rerank=rerank, retrieve_k=retrieve_k,
        )
        all_results.append(results)
        all_profiles.append(profile)

    total_time = time.time() - t0
    log.info(f"/search_with_profile | {total_time:.2f}s | n={len(query)} | rerank={rerank} | {query}")

    return {
        "results": all_results,
        "profiles": all_profiles,
        "total_request_time": total_time,
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "num_gpus": getattr(app.pipeline, "num_gpus", 0),
        "index_backend": getattr(app.pipeline, "index_backend", "cuvs"),
        "num_entries": getattr(app.pipeline, "num_entries", None),
        "num_embedded": getattr(app.pipeline, "num_embedded", None),
    }
