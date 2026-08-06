# Source Code — LLM-Based RAG Pipeline for Open Data Maturity Assessment

I verify that I am the sole author of the programs contained in this archive, except where explicitly stated to the contrary.

[Wong Qian Jie] — [29/7/2026]

## How to run

1. `pip install -r requirements.txt`
2. Create a `.env` file in the **project root** (the same folder as `requirements.txt`, alongside `app/` and `src/`) with:
   ```
   OPENAI_API_KEY=...
   OPENAI_MODEL=...
   EXA_API_KEY=...
   SUPABASE_URL=...
   SUPABASE_KEY=...
   APP_TIMEZONE=Europe/London
   ```
3. Run the pipeline stages in order for evaluation (`src/`): `step1` → `step2` → `step3` → `step4` → `step5` → `step6` → `step7` → `step8_embed_chunks_chroma` → `step9_retrieve_topk_chroma` → `step10_generate_answers_chroma` → `step11_evaluate_chroma` → `step12_combine_evaluation_summaries`.
4. Launch the interactive demo: `streamlit run app/streamlit_app.py`

## Directory contents

### `src/` — final ChromaDB-backed pipeline (used for the reported benchmark results)

| File | Purpose |
|---|---|
| `config.py` | Central configuration: selected question IDs, country file paths, directory constants |
| `utils.py` | Shared helper functions (CSV I/O, directory creation, text cleaning) |
| `step1_build_odm_policy_registry.py` | Builds the registry of the 15 selected ODM questions used for formal evaluation |
| `build_demo_registry_policy_impact.py` | Builds the broader registry of *all* ODM Policy+Impact questions, used by the live UI |
| `step2_build_ground_truth.py` | Constructs the 90-case benchmark ground truth (country × question × answer/score/explanation) |
| `step3_generate_queries.py` | LLM-based multilingual search query generation, including country-specific query guidance (e.g. Estonia) |
| `step4_search_web.py` | Web search via the Exa API, including domain-preference filtering (e.g. Bulgaria) |
| `step5_download_docs.py` | Downloads HTML/PDF sources, blocks answer-leakage and procurement-noise sources |
| `step6_extract_text.py` | Extracts text from HTML (BeautifulSoup) and PDF (pypdf, with PyPDF2 fallback) |
| `step7_chunk_text.py` | Splits extracted text into 180-word chunks with 40-word overlap |
| `step8_embed_chunks_chroma.py` | Generates multilingual embeddings and writes them to the persistent ChromaDB collection |
| `step9_retrieve_topk_chroma.py` | Candidate retrieval, official-source weighting, question-aware reranking, diversification, low-value penalties, targeted refinements (Belgium) |
| `step10_generate_answers_chroma.py` | LLM response/score/justification generation, constrained response selection, second-stage selector |
| `step11_evaluate_chroma.py` | Computes all evaluation metrics against ground truth |
| `step12_combine_evaluation_summaries.py` | Aggregates per-country/per-dimension summaries |
| `pipeline_runner.py` | Orchestrates the full pipeline for a single live query; used by the Streamlit app |

### `src/legacy_baseline/`

| File | Purpose |
|---|---|
| `step8_embed_chunks.py`, `step9_retrieve_topk.py`, `step10_generate_answers.py`, `step11_evaluate.py` | Earlier NumPy/pickle-based retrieval architecture, superseded by the ChromaDB pipeline. Retained because Section 6.3 of the report directly compares this baseline against the final ChromaDB-backed system. Not used by the live UI. |

### `src/` — debug utilities

| File | Purpose |
|---|---|
| `debug_rerun_estonia_only.py` | Standalone script used during error analysis to re-run and inspect Estonia-specific cases |
| `debug_inspect_country_case.py` | Utility for inspecting a single country/question case's retrieval and generation output |

### `app/` — Streamlit interactive demo

| File | Purpose |
|---|---|
| `streamlit_app.py` | Main UI: country/question selection, run pipeline, display results |
| `db.py` | Supabase persistence for saved runs and search history |
| `db_sqlite_backup.py` | Local SQLite fallback persistence, mirroring `db.py`, used when Supabase is unavailable |
| `ui_helpers.py` | Formatting helpers (timestamps, source display, etc.) |
| `export_utils.py` | PDF export of results |
| `country_flags.py` | Country flag display helper |

### `src/tests/` — automated test suites

| File | Purpose |
|---|---|
| `test_pipeline_logic.py` | Unit tests for chunking, ground-truth score inference, response canonicalisation, and evaluation metrics (18 tests) |
| `test_retrieval_reranking.py` | Unit tests for benchmark-leakage blocking, official-source weighting, question-aware reranking, low-value penalties, source diversification, and the Belgium-targeted refinement (27 tests) |
| `test_api_mocked.py` | Mocked tests for query generation and answer generation using `unittest.mock`, verifying prompt construction and fallback behaviour without live API calls (8 tests) |

Run the full suite with:
```bash
python -m pytest src/tests/ -v
```
53 tests total, all passing.

## Notes on architecture

The final ChromaDB-backed pipeline replaces an earlier NumPy/pickle-based implementation, retained under `legacy_baseline/` purely to support the architecture-comparison discussion in Section 6.3 of the report; it is not part of the final system and is not used by the live demo.

ChromaDB is used in two distinct ways. The evaluation pipeline (`step8_embed_chunks_chroma.py`) writes to a disk-backed, persistent ChromaDB collection under `data/interim/chroma_db/`, built once and reused to reproduce the reported benchmark results. The live Streamlit demo (`pipeline_runner.py`) never reads or writes that collection; instead, each live query builds a separate, throwaway, in-memory ChromaDB collection scoped to that single request, consistent with Section 4.5.8 of the report.

## Note on excluded directories

This archive does not include `data/interim/`. This directory holds intermediate pipeline artifacts — most significantly the persistent ChromaDB evidence corpus (`data/interim/chroma_db/`) built during evidence acquisition (for evaluation purposes) — which is excluded because it is large, machine-generated, and fully reproducible from source. Running the pipeline stages listed under "How to run" above regenerates it from scratch; nothing in this folder is hand-authored or required to review the source code itself.

## Pre-built evidence corpus (optional)

`data/interim/` is excluded from this repository (see "Note on excluded
directories" above) because it is large and fully regenerable. For convenience,
a pre-built copy is available here: [Google Drive link: https://drive.google.com/file/d/1LsTMGSIoQGb5fe1yl5jUQcaK4pQbpmSK/view?usp=drive_link]. To use it, download
and extract the zip so its contents sit at `data/interim/` in your local clone,
then skip directly to `step9_retrieve_topk_chroma` and run through
`step12_combine_evaluation_summaries` to view the evaluation results — steps
4–8 (web search through embedding) are the most time-consuming part of the
pipeline and are already baked into the pre-built corpus.