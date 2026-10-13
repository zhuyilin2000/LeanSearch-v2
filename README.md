# LeanSearch v2

> **This is a modified fork.** It is based on
> [frenzymath/LeanSearch-v2](https://github.com/frenzymath/LeanSearch-v2)
> (Apache License 2.0) and is published here under the same license. See
> [Changes from upstream](#changes-from-upstream) at the end of this file for a
> summary of what was modified.

A retrieval system for Lean 4 theorem proving. The corpus is extracted from
Mathlib with [jixia](https://github.com/frenzymath/jixia) and encoded with
[Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B); two modes
sit on top of the same corpus:

- **Standard mode** — single-query embedding + Qwen3-Reranker-8B pipeline.
- **Reasoning mode** — iterative decompose / retrieve / filter / judge loop
  driven by an LLM, targeting *global premise retrieval*.

This repository contains the code, data, and reproduction scripts for the
paper *LeanSearch v2: Global Premise Retrieval for Lean 4 Theorem Proving*
([arXiv:2605.13137](https://arxiv.org/abs/2605.13137)).

Standard mode is also hosted publicly at **<https://leansearch.net/>**:

```bash
curl -X POST https://leansearch.net/search -H 'Content-Type: application/json' \
  -d '{"query": ["the order of a group element divides the order of the group"], "num_results": 10}'
```

Full API reference at <https://leansearch.net/docs>. Note that the public
deployment uses Qwen3-Reranker-4B (a cost-driven variant); the paper's
Table 1 numbers come from the 8B reranker that `./scripts/serve.sh` loads
locally.

## Repository layout

```
config.yaml                  paths, models, retriever URL, LLM, reasoning, prove
config.example.yaml          template for `config.local.yaml` overrides
benchmark/                   MathlibQR.json, MathlibMPR.json, MathlibMPR_Prop_ids.txt,
                             FATE-H.jsonl + FATE-H.LICENSE, MathlibQR_shared171.json
src/leansearchv2/            library code (lightweight to import; GPU deps lazy)
  ├── config.py, llm.py, standard_client.py
  ├── pipeline.py, server.py            standard mode (GPU)
  ├── corpus/                           jixia → JSONL → cuVS pipeline
  ├── reasoning/                        decompose → search → filter → judge loop
  ├── prove/                            prover + verifier + simple reflection loop
  └── eval/                             metric implementations
scripts/                     entry points
external/jixia/              git submodule (pinned to v4.28.0-rc1)
data/                        gitignored: cuVS index, corpus workdir, etc.
```

## Installation

```bash
git clone --recurse-submodules https://github.com/frenzymath/LeanSearch-v2
cd LeanSearch-v2

# Editable install (looser bounds from pyproject.toml):
pip install -e .
pip install -e ".[cuvs]" --extra-index-url https://pypi.nvidia.com   # serving / build_cuvs
pip install -e ".[lean]"                                              # prove task

# Or, for an exact reproduce of the environment used to produce the paper:
pip install -r requirements.txt --extra-index-url https://pypi.nvidia.com
```

Edit `config.yaml` to point at the cuVS index, model checkpoints, retriever
URL, and LLM endpoint. Local overrides go in `config.local.yaml` (gitignored).

## Data

| Artifact | Source |
|---|---|
| Pre-embedded cuVS corpus (Mathlib v4.28.0-rc1, Qwen3-Embedding-8B) | <https://huggingface.co/datasets/FrenzyMath/lsv2-mathlib-v4.28.0-rc1-cuvs> |
| Same corpus as JSONL (one record per declaration) | <https://huggingface.co/datasets/FrenzyMath/lsv2-mathlib-v4.28.0-rc1-jsonl> |
| `benchmark/MathlibQR.json` (200 declarations × 6 query styles = 946 rows) | this repo |
| `benchmark/MathlibQR_shared171.json` (171-decl shared subset → 810 fair query rows) | this repo |
| `benchmark/MathlibMPR.json` (69 theorems, premise-group ground truth) | this repo |
| `benchmark/MathlibMPR_Prop_ids.txt` (50-id Prop subset, ⊆ MathlibMPR) | this repo |
| `benchmark/FATE-H.jsonl` (100 problems) | redistributed under CC BY 4.0; see `benchmark/FATE-H.LICENSE` and cite the FATE paper |

After downloading the cuVS dataset, extract it into `data/cuvs/mathlib-v4.28.0-rc1/`
(or set `paths.cuvs_db` elsewhere).

## Standard mode

### Build the corpus from scratch (optional)

If you want to rebuild from Mathlib source instead of using the pre-built
cuVS dataset, run the four-step pipeline:

```bash
# Prerequisites: Lean 4 toolchain (elan/lake) installed; a Mathlib checkout
# at `paths.mathlib4` with `lake exe cache get` already done; jixia built
# via `cd external/jixia && lake build`; LLM key exported (for step 1c).
./scripts/build_corpus.sh
```

Steps individually:
```bash
python -m leansearchv2.corpus.jixia_extract     # Mathlib *.lean → raw JSON
python -m leansearchv2.corpus.merge_to_jsonl    # raw → merged JSONL (carries refs)
python -m leansearchv2.corpus.informalize       # dep-aware bottom-up informalizer
python -m leansearchv2.corpus.build_cuvs        # JSONL → cuVS index (GPU)
```

The informalizer mirrors §3.1 of the paper: declarations are filtered with
the `is_internal` heuristic (drops compiler-generated `_proof_1` / `match_*`
style names), `typeReferences` (and `valueReferences` for non-Prop kinds)
are resolved against the corpus, the resulting DAG is topologically sorted
with Kahn's algorithm, and informalization runs level-by-level so each
prompt sees its dependencies' already-informalized descriptions. Kind-
specific Jinja templates (`templates/theorem.md.j2`, `definition.md.j2`,
`instance.md.j2`, `technical_entry.md.j2`) drive the prompts. Structural
kinds (`inductive`, `classInductive`, `structure`, `class`, `constructor`,
`recursor`) are informalized too so downstream prompts have context for
them, but only the user-facing kinds (`theorem`, `definition`, `instance`,
`abbrev`, `opaque`, `axiom`) end up in the output JSONL.

The released HuggingFace JSONL was generated by an internal PostgreSQL-
backed variant of this pipeline with stricter resume / cost-tracking
machinery; the algorithm is the same.

### Serve standard mode locally

The embedding model and reranker each require a dedicated GPU, so at least 2 GPUs are needed. With `paths.cuvs_db` and model paths set in `config.yaml`:

```bash
./scripts/serve.sh
```

This starts `leansearchv2.server:app` on `0.0.0.0:8000` (override with
`HOST` / `PORT`). Endpoints: `POST /search`, `POST /search_with_profile`,
`GET /health`. Body params for `/search`:
`query: list[str]`, `num_results: int = 10`, `rerank: bool = True`,
`retrieve_k: int | None = None`.

### CPU-only deployment (SiliconFlow API + PostgreSQL/pgvector)

On machines without NVIDIA GPUs there is a wire-compatible backend that
swaps the whole GPU stack for the SiliconFlow
`Qwen3-Embedding-8B`/`Qwen3-Reranker-8B` APIs plus a pgvector database
(1024-dim MRL truncation; exact-scan retrieval by default, optional HNSW):

```bash
pip install -r requirements-pg.txt      # NOT requirements.txt (needs aarch64-hostile CUDA wheels)
./scripts/serve_pg.sh                   # same endpoints, same wire format
```

See **[docs/deploy-pg.md](docs/deploy-pg.md)** for the full runbook (corpus
download, `load_pg init/load/embed/verify`, cost estimates, evaluation).
Do not install upstream `requirements.txt` on the pg backend — it pins the
cuVS/CUDA stack and requires Python ≥ 3.11.

### Reproduce Table 1 (Search task on MathlibQR)

Start `./scripts/serve.sh` on a GPU pod (the canonical 8B reranker
configuration matches the paper's Table 1). Then:

```bash
python scripts/reproduce_search.py --url http://localhost:8000
```

Outputs `reports/search/{per_query.jsonl, summary.json}`. The script runs
both retriever-only (`rerank=False`) and the full rerank pipeline against
the 946 rows of MathlibQR and prints the fair-subset (810 rows / 171
declarations) and full-subset (946 rows) summary on stdout, reproducing
the LSv2 (retriever-only) and LSv2 (rerank) rows of Table 1.

## Reasoning mode

### Reproduce Table 2 (Global premise retrieval on MathlibMPR)

Reasoning mode talks to the standard-mode endpoint over HTTP. Start
`./scripts/serve.sh`, define one or more LLM profiles under `llm.<name>`
in `config.yaml` (or `config.local.yaml`) with the relevant API key env
var, route the sketch / filter / judge roles via `reasoning.{sketch,filter,judge}_llm`
(paper used `sonnet` for sketch/judge and `kimi` for filter), then:

```bash
python scripts/reproduce_premise.py --url http://localhost:8000
```

Outputs `reports/premise/{per_query.jsonl, summary.json}` and prints
Recall@k(group) and Covered@k (k ∈ {5, 10, 20, 30, 50}) on stdout, matching
the LSv2 (reasoning) row of Table 2.

## Prove task

### Reproduce Table 3 (FATE-H and MathlibMPR-Prop)

Same LLM-profile setup as reasoning mode; route the prover and reflect-time
query generator via `prove.{prover,query}_llm` in `config.yaml` (paper used
`sonnet` for both). Then:

```bash
python scripts/reproduce_prove.py --url http://localhost:8000 \
    --retriever-modes standard,reasoning \
    --benchmark all
```

Outputs `reports/prove/<benchmark>_<mode>/{per_problem.jsonl, summary.json}`
and prints a `#solved` table for FATE-H (100) × {standard, reasoning} and
MathlibMPR-Prop (50) × {standard, reasoning}, matching the LSv2 standard
and LSv2 reasoning rows of Table 3.

Requires `pip install -e ".[lean]"` plus a Lean 4 toolchain (elan/lake)
reachable on `PATH`. The first verification call lazily downloads
the Mathlib cache used by `lean-interact`; expect a one-time delay.

## Citation

```bibtex
@article{leansearchv2,
  title         = {LeanSearch v2: Global Premise Retrieval for Lean 4 Theorem Proving},
  author        = {Gao, Guoxiong and Sun, Zeming and Jiang, Jiedong and Wang, Yutong and
                   Xu, Jingda and Wu, Peihao and Dai, Bryan and Dong, Bin},
  year          = {2026},
  eprint        = {2605.13137},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

If you use FATE-H please also cite

```bibtex
@article{jiang2025fate,
  title={Fate: A formal benchmark series for frontier algebra of multiple difficulty levels},
  author={Jiang, Jiedong and He, Wanyi and Wang, Yuefeng and Gao, Guoxiong and Hu, Yongle and Wang, Jingting and Guan, Nailin and Wu, Peihao and Dai, Chunbo and Xiao, Liang and Dong, Bin},
  journal={arXiv preprint arXiv:2511.02872},
  year={2025}
}
```

## License

Apache 2.0. See `LICENSE`. `benchmark/FATE-H.jsonl` is redistributed under
CC BY 4.0; see `benchmark/FATE-H.LICENSE`. `external/jixia` is licensed
separately (Apache 2.0).

## Changes from upstream

Modifications relative to
[frenzymath/LeanSearch-v2](https://github.com/frenzymath/LeanSearch-v2) `main`
(`94f4888`), as required by Apache License 2.0 §4(b). Every file listed as
*modified* below was changed by this fork.

**Added — a CPU-only backend** (the upstream code requires an NVIDIA GPU: the
`cuVS`/`cupy`/`torch` stack has no wheels for aarch64/NPU hosts). It runs the
same corpus and the same HTTP contract on the SiliconFlow
`Qwen3-Embedding-8B` / `Qwen3-Reranker-8B` APIs plus PostgreSQL/pgvector:

| Path | Purpose |
|---|---|
| `src/leansearchv2/siliconflow.py` | SiliconFlow embeddings/rerank client |
| `src/leansearchv2/pgdb.py` | PostgreSQL connection pool |
| `src/leansearchv2/pg_pipeline.py` | `PGRetrievalPipeline` (same interface as the cuVS pipeline) |
| `src/leansearchv2/corpus/load_pg.py` | corpus loader CLI: `init/load/embed/index/verify/status` |
| `scripts/serve_pg.sh`, `requirements-pg.txt` | launcher and CUDA-free dependency set |
| `docs/deploy-pg.md` | deployment runbook |
| `scripts/check_hnsw_recall.py`, `scripts/check_verifier.py` | verification tools |

`/search`, `/search_with_profile` and `/health` are unchanged in shape, so
existing clients work against either backend; select one with
`serve.index_backend` (`cuvs` or `pg`).

**Modified — bug fixes and hardening** (each reproduced before and after):

- `prove/verifier.py` — a `LeanError` from the REPL was read as a clean compile
  (`getattr(response, "messages", [])` is empty for `LeanError`), so a REPL-level
  failure could be reported as a *successful proof*. Now fails closed.
- `eval/premise_metrics.py` — `all(hit_orig) if orig_groups else True` made
  Covered@k vacuously 1 for rows without original groups (`all([]) == True`).
- `reasoning/decompose.py` — the fenced branch of `_parse_json` skipped the
  `isinstance(obj, dict)` check; query fields are now shape-checked, so a
  returned string no longer explodes into one query per character.
- `reasoning/filter.py` — kept indices were neither de-duplicated nor restored
  to retrieval order.
- `reasoning/run.py` — a failed search silently became "no results".
- `pipeline.py` — `reranker_devices = range(1, num_gpus)` is empty on a
  single-GPU host, and the following `// n` divided by zero *outside* the
  `try`, bypassing the documented fallback.
- `siliconflow.py` — `resp.json()` was called outside the `SiliconFlowError`
  guards on a <400 response, turning a graceful rerank fallback into a 500.
- `config.py` — `bool(get(...))` turned the strings `"0"`/`"false"`/`"no"` into
  `True`; added `parse_bool`/`get_bool`/`get_int`, which raise on unknown tokens.
- `server.py` — `pipeline.close()` was never called; requests are now bounded.
- `standard_client.py` — `INSTRUCTION` moved here so both pipelines share one
  definition.

**Unchanged** — `external/jixia`, `benchmark/*`, the upstream corpus toolchain
(`corpus/build_cuvs.py`, `informalize.py`, `jixia_extract.py`,
`merge_to_jsonl.py`, `templates/*.j2`), the upstream scripts, and the main
logic of `eval/`, `reasoning/`, `prove/`.
