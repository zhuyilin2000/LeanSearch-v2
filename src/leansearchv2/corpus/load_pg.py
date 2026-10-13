"""Corpus loader for the pg backend: JSONL → PostgreSQL/pgvector.

Subcommands (all idempotent / resumable)::

    python -m leansearchv2.corpus.load_pg init     [--jsonl PATH]
    python -m leansearchv2.corpus.load_pg load     [--jsonl PATH] [--batch-rows 1000]
    python -m leansearchv2.corpus.load_pg status
    python -m leansearchv2.corpus.load_pg embed    [--batch-size N] [--concurrency N]
                                                   [--min-interval-ms N]
                                                   [--max-requests N] [--max-tokens N]
                                                   [--price-per-1m-tokens X] [--dry-run]
    python -m leansearchv2.corpus.load_pg index    [--m 16] [--ef-construction 200]
                                                   [--maintenance-work-mem 8GB]
    python -m leansearchv2.corpus.load_pg verify   [--sample 100] [--self-check 20]

`init` creates database ``leansearch`` (in the container pointed at by
``pg.*``), the pgvector extension, and the schema. `load` upserts corpus
metadata (never touching embedding progress). `embed` fills ``embedding``
via the SiliconFlow API with per-batch commits — crash at any point and a
re-run resumes where it stopped. `index` (optional) builds the HNSW index;
without it, retrieval does an exact scan. `verify` checks DB↔JSONL parity
plus an end-to-end self-retrieval sanity test.

Defaults are config-driven: `embed`'s batch size and concurrency come from
``siliconflow.embedding.*``, `index`'s working memory from
``pg.maintenance_work_mem`` (see config.yaml). Every one of them is
overridable by the matching flag or upper-case env var.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

from .. import pgdb
from ..config import get, get_path
from .build_cuvs import construct_lean_representation


log = logging.getLogger("leansearchv2.corpus.load_pg")

EXPECTED_FIELDS = {
    "index", "name", "module_name", "kind", "type", "signature",
    "value", "informal_name", "informal_description",
}

# 3 consecutive failed batches (quota / key class problems) abort the run.
MAX_CONSECUTIVE_BATCH_FAILURES = 3


def _default_jsonl() -> Path:
    return Path(get_path("CORPUS_JSONL", "paths", "corpus_jsonl"))


def _dimensions() -> int:
    return int(get("SILICONFLOW_EMBEDDING_DIMENSIONS", "siliconflow", "embedding", "dimensions", default=1024))


def _ddl(dimensions: int) -> str:
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;

-- NOTE on identity: the JSONL `index` field is NOT unique (it is a local
-- counter from the informalization stage — 310k rows share only ~1k values).
-- The stable natural key is (module_name, name); it is materialized as
-- `src_key` ("module::name", the same doc-id semantics reasoning/prove
-- use) and enforced UNIQUE for idempotent upserts.
CREATE TABLE IF NOT EXISTS entries (
    id                   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    src_key              text NOT NULL UNIQUE,
    corpus_index         bigint NOT NULL,
    name                 text[] NOT NULL,
    module_name          text[] NOT NULL,
    kind                 text NOT NULL,
    "type"               text NOT NULL DEFAULT '',
    signature            text NOT NULL DEFAULT '',
    "value"              text,
    informal_name        text,
    informal_description text,
    text_to_embed        text NOT NULL,
    embedding            vector({dimensions}),
    embedded_at          timestamptz
);

-- Tiny partial index: makes pending-batch claims an index scan, not a scan
-- over the whole embedding column.
CREATE INDEX IF NOT EXISTS entries_pending_idx
    ON entries (id) WHERE embedding IS NULL;

-- Per-batch cost/latency audit (resume + spend tracking across runs).
CREATE TABLE IF NOT EXISTS embed_log (
    id            bigserial PRIMARY KEY,
    ts            timestamptz NOT NULL DEFAULT now(),
    first_id      bigint,
    n_rows        int,
    prompt_tokens bigint,
    total_tokens  bigint,
    elapsed_ms    int
);
"""


# ---------------------------------------------------------------- JSONL input

def _jsonl_rows(path: Path) -> Iterator[tuple[dict[str, Any], str]]:
    """Stream validated (row, text_to_embed) pairs from the corpus JSONL."""
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            def _warn(field: str) -> None:
                print(f"warning: {path}:{lineno}: non-standard {field} literal in JSON, mapped to None", file=sys.stderr)

            def _nan_constant(c: str) -> Any:
                if c in ("NaN", "Infinity", "-Infinity"):
                    _warn(c)
                    return None
                raise ValueError(f"unexpected constant {c}")

            # NOTE: both helpers must be bound *before* json.loads runs. The
            # parse_constant hook is a closure over them, so defining them after
            # this call makes a NaN literal raise NameError from the hook —
            # i.e. it fails in exactly the case it exists to handle.
            try:
                row = json.loads(line, parse_constant=_nan_constant)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {e}")

            if not isinstance(row, dict) or set(row.keys()) != EXPECTED_FIELDS:
                raise SystemExit(
                    f"{path}:{lineno}: field mismatch (got {sorted(row) if isinstance(row, dict) else type(row)})"
                )
            idx = row["index"]
            if not isinstance(idx, int) or isinstance(idx, bool):
                raise SystemExit(f"{path}:{lineno}: `index` is not an int: {idx!r}")
            for field in ("name", "module_name"):
                v = row[field]
                if not isinstance(v, list) or any(not isinstance(x, str) for x in v):
                    raise SystemExit(
                        f"{path}:{lineno}: `{field}` must be a list[str] without nulls "
                        f"(text[] fidelity guarantee): {v!r:.200}"
                    )
            if not isinstance(row["kind"], str) or not row["kind"]:
                raise SystemExit(f"{path}:{lineno}: `kind` must be a non-empty string")
            for field in ("type", "signature", "value", "informal_name", "informal_description"):
                if row[field] is not None and not isinstance(row[field], str):
                    raise SystemExit(f"{path}:{lineno}: `{field}` must be str or null")
            text = construct_lean_representation(row)
            yield row, text


# -------------------------------------------------------------------- init

def cmd_init(args: argparse.Namespace) -> int:
    import psycopg

    cfg = pgdb.conninfo_from_config()
    admin = psycopg.connect(
        host=cfg["host"], port=cfg["port"], dbname="postgres",
        user=cfg["user"], password=cfg["password"], autocommit=True,
    )
    try:
        exists = admin.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (cfg["dbname"],)
        ).fetchone()
        if not exists:
            admin.execute(f'CREATE DATABASE "{cfg["dbname"]}" OWNER "{cfg["user"]}"')
            print(f"created database {cfg['dbname']}")
        else:
            print(f"database {cfg['dbname']} already exists")
    finally:
        admin.close()

    # The vector type must exist before pgdb.connect() can register its
    # adapter, so CREATE EXTENSION goes over a plain connection first.
    conn = psycopg.connect(**cfg)
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        print("extension vector ready")
    finally:
        conn.close()

    conn = pgdb.connect()
    try:
        with conn.transaction():
            conn.execute(_ddl(_dimensions()))
        conn.commit()
        print(f"schema ready (embedding dim = {_dimensions()})")
    finally:
        conn.close()
    return 0


# -------------------------------------------------------------------- load

UPSERT_SQL = """
INSERT INTO entries (src_key, corpus_index, name, module_name, kind, "type",
                     signature, "value", informal_name, informal_description,
                     text_to_embed)
VALUES (%s, %s, %s::text[], %s::text[], %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (src_key) DO UPDATE SET
    corpus_index = EXCLUDED.corpus_index,
    name = EXCLUDED.name, module_name = EXCLUDED.module_name, kind = EXCLUDED.kind,
    "type" = EXCLUDED."type", signature = EXCLUDED.signature, "value" = EXCLUDED."value",
    informal_name = EXCLUDED.informal_name,
    informal_description = EXCLUDED.informal_description,
    text_to_embed = EXCLUDED.text_to_embed
"""


def _src_key(row: dict[str, Any]) -> str:
    return f"{'.'.join(row['module_name'])}::{'.'.join(row['name'])}"


def cmd_load(args: argparse.Namespace) -> int:
    path = Path(args.jsonl) if args.jsonl else _default_jsonl()
    conn = pgdb.connect()
    n = 0
    try:
        buf: list[tuple] = []
        for row, text in _jsonl_rows(path):
            buf.append((
                _src_key(row), row["index"], row["name"], row["module_name"], row["kind"],
                row["type"] or "", row["signature"] or "", row["value"],
                row["informal_name"], row["informal_description"], text,
            ))
            if len(buf) >= args.batch_rows:
                with conn.cursor() as cur:
                    cur.executemany(UPSERT_SQL, buf)
                conn.commit()
                n += len(buf)
                buf = []
        if buf:
            with conn.cursor() as cur:
                cur.executemany(UPSERT_SQL, buf)
            conn.commit()
            n += len(buf)
    finally:
        conn.close()
    print(f"loaded {n} rows into entries (upsert; embeddings untouched)")
    return 0


# ------------------------------------------------------------------- status

def cmd_status(args: argparse.Namespace) -> int:
    conn = pgdb.connect()
    try:
        total, embedded = conn.execute(
            "SELECT count(*), count(embedding) FROM entries"
        ).fetchone()
    finally:
        conn.close()
    total, embedded = int(total), int(embedded)
    pending = total - embedded
    pct = 100.0 * embedded / total if total else 0.0
    print(f"total={total} embedded={embedded} pending={pending} ({pct:.1f}%)")
    return 0


# ------------------------------------------------------------------- embed

CLAIM_SQL = """
SELECT id, text_to_embed FROM entries
WHERE embedding IS NULL ORDER BY id
LIMIT %s FOR UPDATE SKIP LOCKED
"""

UPDATE_SQL = "UPDATE entries SET embedding = %s, embedded_at = now() WHERE id = %s"

LOG_SQL = """
INSERT INTO embed_log (first_id, n_rows, prompt_tokens, total_tokens, elapsed_ms)
VALUES (%s, %s, %s, %s, %s)
"""


def _embed_worker(
    worker_id: int,
    *,
    batch_size: int,
    min_interval_s: float,
    max_requests: int | None,
    max_tokens: int | None,
    price_per_1m: float,
    stop: threading.Event,
    progress: dict,
    progress_lock: threading.Lock,
) -> None:
    from pgvector.psycopg import Vector

    from ..siliconflow import SiliconFlowEmbedder, embedder_from_config

    embedder = embedder_from_config()
    conn = pgdb.connect()
    stop_reason: str | None = None
    consecutive_failures = 0

    def _drain_tokens() -> int:
        """Tokens spent since the last drain, resetting the counter.

        Used on the failure path too: a batch can pay for its prompt tokens
        and then roll back, and a rollback does not refund them.
        """
        spent = embedder.total_prompt_tokens
        embedder.total_prompt_tokens = 0
        return spent

    try:
        while not stop.is_set():
            t0 = time.time()
            try:
                with conn.transaction():
                    rows = conn.execute(CLAIM_SQL, (batch_size,)).fetchall()
                    if not rows:
                        return
                    embeddings = embedder.embed([r[1] for r in rows])
                    with conn.cursor() as cur:
                        cur.executemany(
                            UPDATE_SQL, [(Vector(e), r[0]) for r, e in zip(rows, embeddings)]
                        )
                        cur.execute(
                            LOG_SQL,
                            (
                                rows[0][0], len(rows),
                                embedder.total_prompt_tokens, 0, int((time.time() - t0) * 1000),
                            ),
                        )
                consecutive_failures = 0
            except Exception as e:  # noqa: BLE001 — one bad batch must not kill the run
                consecutive_failures += 1
                # The HTTP call may have succeeded before the DB write failed.
                # Drain those tokens now, or the next successful batch counts
                # them again and the reported cost/ETA drift upward.
                with progress_lock:
                    progress["tokens"] += _drain_tokens()
                log.warning(
                    "[worker %d] batch failed (%d/%d consecutive): %s",
                    worker_id, consecutive_failures, MAX_CONSECUTIVE_BATCH_FAILURES, e,
                )
                if consecutive_failures >= MAX_CONSECUTIVE_BATCH_FAILURES:
                    stop_reason = f"worker {worker_id}: {MAX_CONSECUTIVE_BATCH_FAILURES} consecutive batch failures"
                    stop.set()
                    return
                time.sleep(min(2.0 * consecutive_failures, 30.0))
                continue

            with progress_lock:
                progress["done"] += len(rows)
                progress["tokens"] += _drain_tokens()  # consume per-batch
                progress["requests"] += 1
                done, tokens, requests = progress["done"], progress["tokens"], progress["requests"]
                if max_requests is not None and requests >= max_requests:
                    stop_reason = f"max-requests={max_requests} reached"
                    stop.set()
                if max_tokens is not None and tokens >= max_tokens:
                    stop_reason = f"max-tokens={max_tokens} reached"
                    stop.set()
            if stop_reason:
                log.warning("[worker %d] stopping: %s", worker_id, stop_reason)
                return

            elapsed = time.time() - t0
            if elapsed < min_interval_s:
                time.sleep(min_interval_s - elapsed)

            if worker_id == 0:
                total = progress["total"]
                rate = done / max(progress["elapsed_start"] and time.time() - progress["elapsed_start"], 1e-9)
                eta_min = (total - done) / rate / 60 if rate > 0 else float("nan")
                print(
                    f"done {done}/{total} ({100.0 * done / total:.1f}%) | "
                    f"tokens={tokens:,} | cost≈¥{tokens / 1e6 * price_per_1m:.2f} | "
                    f"rate={rate:.1f} rows/s | ETA≈{eta_min:.0f}min",
                    flush=True,
                )
    finally:
        conn.close()
        embedder.close()
        if stop_reason:
            print(f"[worker {worker_id}] stopped: {stop_reason}", file=sys.stderr)


def cmd_embed(args: argparse.Namespace) -> int:
    conn = pgdb.connect()
    try:
        total, embedded = conn.execute(
            "SELECT count(*), count(embedding) FROM entries"
        ).fetchone()
    finally:
        conn.close()
    total, embedded = int(total), int(embedded)
    pending = total - embedded
    print(f"pending rows: {pending} (total {total}, already embedded {embedded})")
    if pending == 0:
        print("nothing to do")
        return 0

    if args.dry_run:
        # ~3.5 chars per token heuristic on text_to_embed.
        conn = pgdb.connect()
        try:
            chars = conn.execute(
                "SELECT sum(length(text_to_embed)) FROM entries WHERE embedding IS NULL"
            ).fetchone()[0] or 0
        finally:
            conn.close()
        est_tokens = int(chars / 3.5)
        print(
            f"dry-run: {pending} rows, {int(chars):,} chars, ≈{est_tokens:,} tokens, "
            f"cost≈¥{est_tokens / 1e6 * args.price_per_1m_tokens:.2f} "
            f"(at ¥{args.price_per_1m_tokens}/1M tokens)"
        )
        return 0

    stop = threading.Event()
    progress = {
        "total": pending, "done": 0, "tokens": 0, "requests": 0,
        "elapsed_start": time.time(),
    }
    progress_lock = threading.Lock()
    threads = [
        threading.Thread(
            target=_embed_worker,
            args=(w,),
            kwargs=dict(
                batch_size=args.batch_size,
                min_interval_s=args.min_interval_ms / 1000.0,
                max_requests=args.max_requests,
                max_tokens=args.max_tokens,
                price_per_1m=args.price_per_1m_tokens,
                stop=stop, progress=progress, progress_lock=progress_lock,
            ),
            daemon=True,
        )
        for w in range(args.concurrency)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    conn = pgdb.connect()
    try:
        total2, embedded2 = conn.execute(
            "SELECT count(*), count(embedding) FROM entries"
        ).fetchone()
    finally:
        conn.close()
    skipped = pending - (int(embedded2) - embedded)
    print(
        f"embedded {int(embedded2) - embedded} rows this run; "
        f"now {int(embedded2)}/{total2} total"
        + (f"; {skipped} rows skipped/failed (still NULL, re-run to retry)" if skipped > 0 else "")
    )
    return 0 if int(embedded2) == total2 else 1


# ------------------------------------------------------------------- index

def cmd_index(args: argparse.Namespace) -> int:
    conn = pgdb.connect()
    try:
        print(
            "building HNSW index (this is optional; exact scan works without it; "
            f"maintenance_work_mem = {args.maintenance_work_mem})..."
        )
        t0 = time.time()
        with conn.transaction():
            # maintenance_work_mem must be allocated from *private* memory here.
            # With parallel maintenance workers Postgres carves it out of a
            # dynamic shared memory segment (dynamic_shared_memory_type=posix ->
            # /dev/shm), which a stock Docker container caps at 64MB: an 8GB
            # build then dies with
            #   DiskFull: could not resize shared memory segment ... No space left
            # even though the data directory has plenty of room. Serial builds
            # use the backend's own address space instead, which is what the
            # measurements in the comment below were taken with.
            conn.execute(
                "SELECT set_config('max_parallel_maintenance_workers', %s, false)",
                (str(args.parallel_maintenance_workers),),
            )
            # HNSW builds its graph in memory and spills to disk when this is
            # too small, which roughly triples build time (measured on this
            # corpus: 20k x 1024-dim vectors spill below ~96MB, i.e. ~4KB per
            # row, so the full 310k-row build needs >= ~1.4GB). set_config()
            # rather than SET so the value is a bound parameter that Postgres
            # parses and validates — a bad value fails loudly here.
            conn.execute(
                "SELECT set_config('maintenance_work_mem', %s, false)",
                (args.maintenance_work_mem,),
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS entries_embedding_hnsw "
                f"ON entries USING hnsw (embedding vector_cosine_ops) "
                f"WITH (m = {args.m}, ef_construction = {args.ef_construction})"
            )
        conn.execute("ANALYZE entries")
        conn.commit()
        print(f"HNSW index built in {time.time() - t0:.0f}s; ANALYZE done")
    finally:
        conn.close()
    return 0


# ------------------------------------------------------------------ verify

def cmd_verify(args: argparse.Namespace) -> int:
    from pgvector.psycopg import Vector

    from ..siliconflow import embedder_from_config

    path = Path(args.jsonl) if args.jsonl else _default_jsonl()
    failures: list[str] = []

    # 1) Load JSONL side: src_key -> row dict (for set parity + field round-trip).
    jsonl_rows: dict[str, dict[str, Any]] = {}
    for row, _ in _jsonl_rows(path):
        jsonl_rows[_src_key(row)] = row
    print(f"JSONL rows: {len(jsonl_rows)}")

    conn = pgdb.connect()
    try:
        total, embedded = conn.execute(
            "SELECT count(*), count(embedding) FROM entries"
        ).fetchone()
        total, embedded = int(total), int(embedded)
        print(f"DB rows: {total}, embedded: {embedded}")
        if total != len(jsonl_rows):
            failures.append(f"row count mismatch: DB {total} vs JSONL {len(jsonl_rows)}")
        if embedded != total:
            failures.append(f"{total - embedded} rows still have NULL embedding")

        # 2) src_key-set parity
        db_keys = {str(r[0]) for r in conn.execute("SELECT src_key FROM entries").fetchall()}
        if db_keys != set(jsonl_rows.keys()):
            only_db = db_keys - set(jsonl_rows.keys())
            only_jsonl = set(jsonl_rows.keys()) - db_keys
            failures.append(f"src_key set mismatch: only-in-DB {len(only_db)}, only-in-JSONL {len(only_jsonl)}")

        # 3) full field round-trip (streaming, ordered by src_key)
        mismatches = 0
        with conn.cursor(name="verify_roundtrip") as cur:  # server-side cursor
            cur.itersize = 2000
            cur.execute(
                "SELECT src_key, corpus_index, name, module_name, kind, \"type\", signature, "
                "\"value\", informal_name, informal_description, text_to_embed "
                "FROM entries ORDER BY src_key"
            )
            for (src_key, corpus_index, name, module_name, kind, type_, signature, value,
                 informal_name, informal_description, text) in cur:
                row = jsonl_rows.get(src_key)
                if row is None:
                    continue
                expected_text = construct_lean_representation(row)
                if (
                    corpus_index != row["index"]
                    or name != row["name"] or module_name != row["module_name"]
                    or kind != row["kind"] or type_ != (row["type"] or "")
                    or signature != (row["signature"] or "")
                    or value != row["value"] or informal_name != row["informal_name"]
                    or informal_description != row["informal_description"]
                    or text != expected_text
                ):
                    mismatches += 1
                    if mismatches <= 5:
                        failures.append(f"row {src_key}: field round-trip mismatch")
        if mismatches > 5:
            failures.append(f"... and {mismatches - 5} more round-trip mismatches")
        print(f"field round-trip: {mismatches} mismatches over {total} rows")

        # 4) embedding norms (sample)
        bad_norms = 0
        for (emb,) in conn.execute(
            "SELECT embedding FROM entries ORDER BY random() LIMIT %s", (args.sample,)
        ).fetchall():
            if emb is None or abs(math.sqrt(math.fsum(x * x for x in emb)) - 1.0) > 1e-3:
                bad_norms += 1
        if bad_norms:
            failures.append(f"{bad_norms}/{args.sample} sampled embeddings not unit-norm")
        print(f"norm check: {bad_norms} bad of {args.sample} sampled")

        # 5) self-retrieval sanity: embed a benchmark entry's own text, it
        #    should rank itself within the top-2 of top-5.
        bench_path = Path(__file__).resolve().parents[3] / "benchmark" / "MathlibQR.json"
        if bench_path.exists() and args.self_check > 0:
            bench = json.loads(bench_path.read_text())
            full_names = [b["full_name"] for b in bench]
            embedder = embedder_from_config()
            try:
                checked = 0
                bad_self = 0
                for fn in full_names:
                    if checked >= args.self_check:
                        break
                    row = conn.execute(
                        "SELECT id, text_to_embed FROM entries "
                        "WHERE array_to_string(name, '.') = %s LIMIT 1", (fn,)
                    ).fetchone()
                    if row is None:
                        continue
                    checked += 1
                    emb = embedder.embed_one(row[1])
                    top5 = conn.execute(
                        "SELECT array_to_string(name, '.') FROM entries "
                        "ORDER BY embedding <=> %s LIMIT 5", (Vector(emb),)
                    ).fetchall()
                    names5 = [t[0] for t in top5]
                    rank = names5.index(fn) + 1 if fn in names5 else 99
                    if rank > 2:
                        bad_self += 1
                        print(f"  self-check: {fn} ranked {rank} (top5={names5[:3]}...)")
                if bad_self:
                    failures.append(f"{bad_self}/{checked} self-retrieval sanity failures (rank > 2)")
                print(f"self-retrieval: {checked - bad_self}/{checked} ok")
            finally:
                embedder.close()
    finally:
        conn.close()

    if failures:
        print("\nVERIFY FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nVERIFY OK")
    return 0


# --------------------------------------------------------------------- main

def main() -> int:
    import dotenv

    dotenv.load_dotenv()  # .env at repo root holds SILICONFLOW_API_KEY / PG password

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="create database + extension + schema")
    p.add_argument("--jsonl", type=str, default=None)

    p = sub.add_parser("load", help="upsert corpus metadata from JSONL")
    p.add_argument("--jsonl", type=str, default=None)
    p.add_argument("--batch-rows", type=int, default=1000)

    sub.add_parser("status", help="print total / embedded / pending counts")

    p = sub.add_parser("embed", help="embed pending rows via SiliconFlow API (resumable)")
    p.add_argument("--batch-size", type=int,
                   default=int(get("SILICONFLOW_EMBEDDING_BATCH_SIZE", "siliconflow", "embedding", "batch_size", default=32)))
    p.add_argument("--concurrency", type=int,
                   default=int(get("SILICONFLOW_EMBEDDING_CONCURRENCY", "siliconflow", "embedding", "concurrency", default=1)))
    p.add_argument("--min-interval-ms", type=int, default=250)
    p.add_argument("--max-requests", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--price-per-1m-tokens", type=float,
                   default=float(get("SILICONFLOW_EMBEDDING_PRICE_PER_1M_TOKENS",
                                     "siliconflow", "embedding", "price_per_1m_tokens", default=2.0)),
                   help="only affects the cost estimate; Qwen3-Embedding-8B list price is ¥2/1M tokens")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("index", help="build optional HNSW index")
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--maintenance-work-mem", type=str,
                   default=get("PG_MAINTENANCE_WORK_MEM", "pg", "maintenance_work_mem", default="8GB"),
                   help="HNSW build memory; ~4KB per row is enough to avoid spilling")
    p.add_argument("--parallel-maintenance-workers", type=int,
                   default=int(get("PG_PARALLEL_MAINTENANCE_WORKERS", "pg",
                                   "parallel_maintenance_workers", default=0)),
                   help="0 (default) keeps the build serial so maintenance_work_mem comes "
                        "from private memory; >0 needs a large /dev/shm in the container")

    p = sub.add_parser("verify", help="check DB↔JSONL parity + self-retrieval sanity")
    p.add_argument("--jsonl", type=str, default=None)
    p.add_argument("--sample", type=int, default=100)
    p.add_argument("--self-check", type=int, default=20)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per API call is noise

    return {
        "init": cmd_init,
        "load": cmd_load,
        "status": cmd_status,
        "embed": cmd_embed,
        "index": cmd_index,
        "verify": cmd_verify,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
