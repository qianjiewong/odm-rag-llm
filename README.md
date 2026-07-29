# Source Code — LLM-Based RAG Pipeline for Open Data Maturity Assessment

I verify that I am the sole author of the programs contained in this archive, except where explicitly stated to the contrary.

[Wong Qian Jie] — [29/7/2026]

## How to run

1. `pip install -r requirements.txt`
2. Create a `.env` file in `src/` with:
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

### `src/legacy_baseline/` *(if reorganised — see note below)*

| File | Purpose |
|---|---|
| `step8_embed_chunks.py`, `step9_retrieve_topk.py`, `step10_generate_answers.py`, `step11_evaluate.py` | Earlier NumPy/pickle-based retrieval architecture, superseded by the ChromaDB pipeline. Retained because Section 6.3 of the report directly compares this baseline against the final ChromaDB-backed system. Not used by the live UI. |

### `src/experiments/` *(if reorganised)*

| File | Purpose |
|---|---|
| `step10_generate_answers_no_rag.py`, `step11_evaluate_no_rag.py` | Earlier RAG-vs-non-RAG ablation, generating answers without retrieved evidence for comparison purposes. |

### `src/debug/` *(if reorganised)*

| File | Purpose |
|---|---|
| `debug_rerun_estonia_only.py` | Standalone script used during error analysis to re-run and inspect Estonia-specific cases |
| `debug_inspect_country_case.py` | Utility for inspecting a single country/question case's retrieval and generation output |

### `app/` — Streamlit interactive demo

| File | Purpose |
|---|---|
| `streamlit_app.py` | Main UI: country/question selection, run pipeline, display results |
| `db.py` | Supabase persistence for saved runs and search history |
| `ui_helpers.py` | Formatting helpers (timestamps, source display, etc.) |
| `export_utils.py` | PDF export of results |
| `country_flags.py` | Country flag display helper |

## Notes on architecture

The final system uses ChromaDB as the persistent vector database. An earlier NumPy/pickle-based implementation is retained under `legacy_baseline/` purely to support the architecture-comparison discussion in Section 6.3 of the report; it is not part of the final system and is not used by the live demo.