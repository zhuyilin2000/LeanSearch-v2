#!/usr/bin/env bash
# Start the LeanSearch v2 standard-mode FastAPI server on the pg backend
# (SiliconFlow API embeddings/rerank + PostgreSQL/pgvector; CPU-only).
#
# Requires: .env at the repo root with SILICONFLOW_API_KEY and
# LEANSEARCH_PG_PASSWORD (see docs/deploy-pg.md), plus a populated
# `leansearch` database (see leansearchv2.corpus.load_pg).
#
# Tuning: PG_*/SILICONFLOW_* env vars or config.local.yaml (see config.yaml).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

if [ ! -f .env ]; then
    echo "error: $REPO_DIR/.env not found (needs SILICONFLOW_API_KEY / LEANSEARCH_PG_PASSWORD)" >&2
    exit 1
fi
if ! grep -q '^SILICONFLOW_API_KEY=..' .env; then
    echo "error: SILICONFLOW_API_KEY missing/empty in .env" >&2
    exit 1
fi

# PYTHONPATH is overwritten (not appended): the host's Ascend toolkit exports
# its own site-packages globally, which can shadow this repo's dependencies.
export PYTHONPATH="$REPO_DIR/src"
export INDEX_BACKEND="${INDEX_BACKEND:-pg}"

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

if [ -x "$REPO_DIR/.venv/bin/python" ]; then
    PY="$REPO_DIR/.venv/bin/python"
else
    PY="python"
fi

exec "$PY" -m uvicorn leansearchv2.server:app --host "$HOST" --port "$PORT"
