# CPU-only deployment: SiliconFlow API + PostgreSQL/pgvector

This runbook deploys LeanSearch v2's standard-mode service on a machine with
**no NVIDIA GPUs** (e.g. the 192-core aarch64 box with Ascend NPUs): embeddings
and reranking go through the [SiliconFlow](https://siliconflow.cn) API, vector
search runs in PostgreSQL/pgvector, and the server is pure CPU.

What you get: a wire-compatible replacement of `scripts/serve.sh`'s service —
`POST /search`, `POST /search_with_profile`, `GET /health` — that
`standard_client.StandardClient`, reasoning mode, the prove task, and
`scripts/reproduce_search.py` consume unchanged.

> Do **not** install the upstream `requirements.txt` here: `cuvs-cu12` /
> `cupy-cuda12x` have no aarch64 wheels, and `pyproject.toml` requires
> Python ≥ 3.11 while the host has 3.10. Use `requirements-pg.txt` and the
> `PYTHONPATH=src` pattern (the `serve_pg.sh` script does this for you).

## 0. Prerequisites

- Python 3.10 venv (`python3 -m venv .venv`)
- A PostgreSQL 17 instance with the [pgvector](https://github.com/pgvector/pgvector)
  0.7+ extension available (e.g. the `pgvector/pgvector:pg17` docker image)
- A SiliconFlow API key with `Qwen/Qwen3-Embedding-8B` (paid) and
  `Qwen/Qwen3-Reranker-8B` access
- Network access to `api.siliconflow.cn`; corpus download via
  `hf-mirror.com` if huggingface.co is unreachable

Setup:

```bash
cd LeanSearch-v2
python3 -m venv .venv
.venv/bin/pip install -r requirements-pg.txt

# secrets (gitignored)
cat > .env <<'EOF'
SILICONFLOW_API_KEY=sk-...
LEANSEARCH_PG_PASSWORD=<pg password>
EOF
chmod 600 .env

# local config overrides (gitignored)
cat > config.local.yaml <<'EOF'
serve:
  index_backend: pg
retriever:
  timeout_s: 120   # eval reranks retrieve_k=200 docs via API
EOF
```

Connection settings live under `pg:` in `config.yaml` (host/port/dbname/user);
each is overridable by `PG_HOST` / `PG_PORT` / `PG_DBNAME` / `PG_USER` /
`PG_PASSWORD` env vars.

## 1. Corpus

Download the pre-built corpus (Mathlib v4.28.0-rc1, informal descriptions
included — this is what the paper's Table 1 numbers were produced from):

```bash
mkdir -p data/jsonl
curl -L -C - -o data/jsonl/lsv2-mathlib-v4.28.0-rc1.jsonl \
  "https://hf-mirror.com/datasets/FrenzyMath/lsv2-mathlib-v4.28.0-rc1-jsonl/resolve/main/lsv2-mathlib-v4.28.0-rc1.jsonl"
```

(Use `huggingface.co` instead of `hf-mirror.com` if it is reachable.)
~332 MB, 310,579 rows. Note the JSONL `index` field is **not** unique — it's
a local counter from the informalization stage — so the loader keys rows by
`src_key = "{module}::{name}"` (which *is* unique, verified).

## 2. Load + embed (one-time, ~1 h and a few ¥)

```bash
export PYTHONPATH=src

.venv/bin/python -m leansearchv2.corpus.load_pg init      # db + extension + schema (idempotent)
.venv/bin/python -m leansearchv2.corpus.load_pg load      # metadata upsert (~2 min)
.venv/bin/python -m leansearchv2.corpus.load_pg embed --dry-run --price-per-1m-tokens 2.0
.venv/bin/python -m leansearchv2.corpus.load_pg status
```

`embed` spends real money (~70M tokens; ≈¥35 at ¥0.5/1M, ≈¥140 at ¥2/1M —
check your tier). It is fully resumable: progress is committed per API batch,
rows keep `embedding IS NULL` until done, and a re-run resumes where it
stopped. Guardrails: `--max-requests N`, `--max-tokens N`; abort after 3
consecutive failed batches (e.g. quota exhausted) with progress preserved.

Recommended invocation (detached, ~50–60 min at 115 rows/s):

```bash
mkdir -p data/logs
setsid nohup env PYTHONPATH=src .venv/bin/python -m leansearchv2.corpus.load_pg embed \
  --concurrency 4 --price-per-1m-tokens 2.0 \
  > data/logs/embed.log 2>&1 &
grep '^done' data/logs/embed.log | tail    # progress lines
```

Afterwards:

```bash
.venv/bin/python -m leansearchv2.corpus.load_pg verify   # DB↔JSONL parity + self-retrieval sanity
.venv/bin/python -m leansearchv2.corpus.load_pg status   # expect: pending=0
```

## 3. Retrieval strategy: exact scan (default) vs HNSW

With 310k × 1024-dim vectors (~600 MB, page-cached), `ORDER BY embedding <=> $q`
**without an index** is an exact scan: ~150–500 ms per query, recall 1.0
(no worse than the paper's approximate cuVS CAGRA). Request latency is
dominated by the two SiliconFlow API calls anyway, so the first deployment
skips the ANN index entirely — zero tuning, zero recall loss.

If you later want sub-10 ms retrieval, add the index (no code changes; the
planner picks it up automatically, and each pooled connection already sets
`hnsw.ef_search` from `pg.ef_search`, default 512 — the stock default of 40
would silently miss top-200 candidates):

```bash
.venv/bin/python -m leansearchv2.corpus.load_pg index   # optional HNSW (m=16, ef_construction=200)
```

`pg.exact_scan: true` (or env `PG_EXACT_SCAN=1`) forces a seq scan again for
A/B comparisons.

## 4. Serve

```bash
./scripts/serve_pg.sh            # 0.0.0.0:8000; HOST/PORT env vars override
```

Sanity:

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"query": ["the order of a group element divides the order of the group"], "num_results": 10}'
```

Smoke-test findings from this deployment (2026-10, recorded for reuse):
`dimensions=1024` honored; batch 512 OK (we use 128); rerank 200 docs / 0.1 MB
payload OK in ~6 s; no 429s at concurrency 4; API vectors already unit-norm;
**`instruction` field supported and effective** on `/v1/rerank` (the pipeline
passes v2's INSTRUCTION, matching the paper's rerank prompt).

## 5. Evaluate (Table 1 reproduction)

```bash
PYTHONPATH=src .venv/bin/python scripts/reproduce_search.py --url http://localhost:8000
```

Prints fair-subset (810 rows / 171 declarations) and full (946 rows) nDCG@k /
Recall@k for retriever-only and rerank configs. Expected deviation vs the
paper's Table 1: retriever-only slightly below (1024-dim MRL truncation vs the
native 4096-dim build), rerank row closer (same 8B reranker model, custom
instruction honored). See `reports/search/summary.json` for the raw numbers.

### Findings from this deployment (2026-10-13)

Full 946-query eval over the complete 310,579-entry corpus (~1.2 h wall clock,
~1.3 s/query retriever-only, ~4.3 s/query rerank — SiliconFlow API latency
dominates):

| slice | config | n | nDCG@1 | nDCG@5 | nDCG@10 | R@1 | R@5 | R@10 | R@50 | R@100 |
|---|---|---|---|---|---|---|---|---|---|---|
| fair | retriever-only | 810 | 0.341 | 0.480 | 0.505 | 0.341 | 0.601 | 0.678 | 0.795 | 0.827 |
| fair | **rerank** | 810 | **0.465** | **0.594** | **0.617** | **0.465** | **0.705** | **0.777** | 0.841 | 0.856 |
| full | retriever-only | 946 | 0.320 | 0.455 | 0.478 | 0.320 | 0.572 | 0.644 | 0.757 | 0.784 |
| full | **rerank** | 946 | **0.438** | **0.564** | **0.585** | **0.438** | **0.673** | **0.737** | 0.797 | 0.810 |

Interpretation against the paper's Table 1 (exact figures in
[arXiv:2605.13137](https://arxiv.org/abs/2605.13137); the deployment-acceptance
targets set for this run were fair-subset rerank R@10 ≥ 0.75 and
retriever-only in the 0.55–0.65 band):

- **rerank R@10 = 0.777 (fair) / 0.737 (full)** — meets the ≥0.75 acceptance
  target on the fair subset; the 8B reranker with the custom `instruction`
  behaves as the paper's.
- **retriever-only R@10 = 0.678 (fair)** — slightly *above* the 0.55–0.65
  band we budgeted for 1024-dim MRL truncation; the exact-scan retrieval
  (recall 1.0, vs the paper's approximate cuVS CAGRA) likely offsets the
  truncation loss.
- rerank lifts R@1 by ~36% relative (0.341 → 0.465 fair), consistent with
  the paper's reranker delta.

One-time corpus cost: 71.3M prompt tokens (batch-128, concurrency 4, ~65 min);
at SiliconFlow's Qwen3-Embedding-8B tier this ran to roughly ¥35–140
depending on price/M — check the console billing for the exact figure.

## 6. Ops notes

- Config is read once per process (`config._CACHE`): **restart** the server
  and the loader after changing `config.yaml` / `config.local.yaml`.
- Changing the embedding dimension later means re-embedding everything
  (drop the `entries` table, `init`, `load`, `embed`) — ¥35–140 and an hour.
- `load_pg load` re-runs are safe (upsert by `src_key`; embeddings preserved)
  as long as it is the **same JSONL**. Pointing it at a different corpus
  would leave stale embeddings behind — drop the table instead.
- Spend audit: `SELECT sum(prompt_tokens) FROM embed_log;`
- The rerank call degrades to retriever order on API failure (logged as a
  warning) — the same fallback semantics as the cuVS pipeline.
- This host's Ascend toolkit exports a global `PYTHONPATH` with its own
  site-packages; `serve_pg.sh` **overwrites** `PYTHONPATH` (rather than
  appending) to keep the venv's dependencies authoritative. Runbook
  commands in this document use `env PYTHONPATH=src` for the same reason.
