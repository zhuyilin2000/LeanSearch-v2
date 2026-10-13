#!/usr/bin/env python
"""Measure pgvector HNSW recall against the exact scan, plus both latencies.

`load_pg verify` only checks self-retrieval identity; it never compares the
approximate index against the exact ordering, so it cannot tell you whether
`hnsw.ef_search` is set high enough. This script does that comparison directly:

    for each probe vector
        exact = ORDER BY embedding <=> v LIMIT k   with enable_indexscan = off
        ann   = ORDER BY embedding <=> v LIMIT k   using the HNSW index
        recall = |exact & ann| / k

Probes are real corpus vectors sampled at random, so both arms see the same
query vector and the only variable is the scan strategy.

    PYTHONPATH=src ./.venv/bin/python scripts/check_hnsw_recall.py
    PYTHONPATH=src ./.venv/bin/python scripts/check_hnsw_recall.py --probes 30 --k 50 200

Run it before building the index to record a baseline; without an HNSW index
both arms are exact scans and every recall is 1.0.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dotenv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
dotenv.load_dotenv(REPO_ROOT / ".env")

from leansearchv2 import pgdb  # noqa: E402
from leansearchv2.config import get_int  # noqa: E402

SQL = "SELECT id FROM entries ORDER BY embedding <=> %s LIMIT %s"


def _has_hnsw(conn) -> bool:
    return bool(
        conn.execute(
            "SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
            "WHERE c.relname = 'entries_embedding_hnsw' AND i.indisvalid"
        ).fetchone()
    )


def _time_query(conn, vec, k: int, *, exact: bool) -> tuple[list[int], float]:
    """Run one query and return (ids, ms).

    `enable_indexscan` is set *symmetrically* with `set_config(..., true)`:
    `SET LOCAL enable_indexscan = off` from the exact arm would otherwise survive
    into the following queries — the connection stays inside one implicit
    transaction, and transaction-scoped GUCs are not undone by the inner
    savepoint. That made every "HNSW" measurement a sequential scan.
    """
    with conn.transaction():
        conn.execute(
            "SELECT set_config('enable_indexscan', %s, true)",
            ("off" if exact else "on",),
        )
        t0 = time.perf_counter()
        rows = conn.execute(SQL, (vec, k)).fetchall()
        elapsed = (time.perf_counter() - t0) * 1000.0
    return [r[0] for r in rows], elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probes", type=int, default=20, help="number of sampled query vectors")
    ap.add_argument("--k", type=int, nargs="+", default=[10, 50, 200], help="cutoffs to compare")
    ap.add_argument("--ef-search", type=int, nargs="+", default=None,
                    help="ef_search values to sweep (default: the configured one)")
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling seed; 0 = spread probes evenly across the table")
    args = ap.parse_args()

    conn = pgdb.connect(autocommit=True)
    try:
        total = conn.execute("SELECT count(*) FROM entries WHERE embedding IS NOT NULL").fetchone()[0]
        indexed = _has_hnsw(conn)
        configured_ef = get_int("PG_EF_SEARCH", "pg", "ef_search", default=512)
        ef_values = args.ef_search or [configured_ef]

        print(f"  rows with embeddings : {total:,}")
        print(f"  HNSW index present   : {indexed}")
        print(f"  probes / k           : {args.probes} / {args.k}")
        print(f"  ef_search sweep      : {ef_values}")
        if not indexed:
            print("\n  NOTE: no HNSW index — both arms are exact scans, recall is trivially 1.0.")

        # Sample probe vectors. `id` ordering is stable, so an even stride gives a
        # deterministic spread without ORDER BY random() over 310k rows.
        stride = max(1, total // args.probes)
        probe_rows = [
            r[0]
            for r in conn.execute(
                "SELECT embedding FROM entries WHERE embedding IS NOT NULL AND id %% %s = 0 "
                "ORDER BY id LIMIT %s",
                (stride, args.probes),
            ).fetchall()
        ]
        if not probe_rows:
            probe_rows = [
                r[0]
                for r in conn.execute(
                    "SELECT embedding FROM entries WHERE embedding IS NOT NULL LIMIT %s",
                    (args.probes,),
                ).fetchall()
            ]
        print(f"  sampled probes       : {len(probe_rows)}\n")

        # Exact arm: one pass per k, reused across all ef values.
        exact: dict[int, list[list[int]]] = {}
        exact_ms: dict[int, list[float]] = {}
        for k in args.k:
            hits, times = [], []
            for vec in probe_rows:
                ids, ms = _time_query(conn, vec, k, exact=True)
                hits.append(ids)
                times.append(ms)
            exact[k] = hits
            exact_ms[k] = times
            print(f"  exact  k={k:<4} {statistics.median(times):8.1f} ms/query (median)")

        print()
        for ef in ef_values:
            conn.execute(f"SET hnsw.ef_search = {int(ef)}")
            print(f"  --- HNSW ef_search={ef} ---")
            for k in args.k:
                recalls, times, returned = [], [], []
                for vec, want in zip(probe_rows, exact[k]):
                    ids, ms = _time_query(conn, vec, k, exact=False)
                    recalls.append(len(set(ids) & set(want)) / max(len(want), 1))
                    times.append(ms)
                    returned.append(len(ids))
                r = statistics.mean(recalls)
                lo = min(recalls)
                # ef_search < k makes HNSW return fewer rows than asked for — a
                # silent truncation, not merely a recall dip.
                trunc = "" if min(returned) >= k else f"  !! only {min(returned)}/{k} rows returned"
                print(
                    f"    k={k:<4} recall {r*100:6.1f}%  (min {lo*100:5.1f}%)   "
                    f"{statistics.median(times):7.1f} ms/query   "
                    f"speedup {statistics.median(exact_ms[k]) / max(statistics.median(times), 0.001):6.1f}x{trunc}"
                )
            print()

        if indexed and len(ef_values) > 1:
            print("  Pick the smallest ef_search that keeps recall at 100% at your largest k.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
