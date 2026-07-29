from __future__ import annotations

from pathlib import Path

import pandas as pd
import chromadb
from chromadb.config import Settings

from config import PROCESSED_DIR, UI_RUNTIME_DIR
from step3_generate_queries import generate_queries_for_question
from step4_search_web import search_web_for_question
from step5_download_docs import download_docs_for_question
from step6_extract_text import extract_text_for_question
from step7_chunk_text import chunk_text_for_question
from step8_embed_chunks_chroma import (
    prepare_chunk_texts,
    generate_embeddings,
    build_chroma_records,
)
from step9_retrieve_topk_chroma import (
    retrieve_topk_for_question_chroma,
    get_retrieval_model,
)
from step10_generate_answers_chroma import generate_answer_for_question, get_openai_client

RUNTIME_COLLECTION_NAME = "odm_chunks_runtime"


def safe_text(value) -> str:
    return str(value or "").strip()


def normalize_whitespace(text: str) -> str:
    return " ".join(safe_text(text).split())


def canonicalize_final_answer(answer: str, response_scoring=None) -> str:
    """
    Convert model output labels into canonical benchmark-style form.

    This mirrors the Step 10 canonicalization so the batch pipeline always
    writes consistent labels into the final dataframe.
    """
    raw = normalize_whitespace(answer)
    if not raw:
        return raw

    candidate_options = []

    if isinstance(response_scoring, dict):
        candidate_options = [normalize_whitespace(k) for k in response_scoring.keys()]
    elif isinstance(response_scoring, list):
        candidate_options = [normalize_whitespace(x) for x in response_scoring]

    raw_lower = raw.lower()

    for option in candidate_options:
        if raw_lower == option.lower():
            return option

    canonical_map = {
        "yes": "Yes",
        "no": "No",
        "partially": "Partially",
        "partial": "Partially",
        "not applicable": "Not applicable",
        "n/a": "Not applicable",
        "unknown": "Unknown",
        "other": "Other",
        "all public bodies": "All public bodies",
        "the majority of public bodies": "The majority of public bodies",
        "a minority of public bodies": "A minority of public bodies",
        "none of the public bodies": "None of the public bodies",
    }

    if raw_lower in canonical_map:
        return canonical_map[raw_lower]

    if len(raw.split()) <= 4:
        return raw[:1].upper() + raw[1:]

    return raw


def build_runtime_chroma_collection(chunk_df: pd.DataFrame, model):
    """
    Embed freshly-scraped runtime evidence and insert it into a throwaway,
    in-memory Chroma collection. This is the only evidence store the live
    demo ever queries. It deliberately never touches the persistent
    chroma_db/ collection used to produce the reported benchmark results,
    so that a deployed demo always reflects the current state of the web
    rather than the evidence snapshot the evaluation was run against.
    """
    if chunk_df.empty:
        return None

    texts = prepare_chunk_texts(chunk_df)
    embeddings = generate_embeddings(model=model, texts=texts, show_progress_bar=False)

    # In-memory client (no path=...) -> nothing is written to disk and nothing
    # persists between calls. A fresh collection per call keeps runs isolated.
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    collection = client.create_collection(name=RUNTIME_COLLECTION_NAME)

    ids, documents, metadatas, embedding_list = build_chroma_records(chunk_df, embeddings)
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embedding_list,
    )

    return collection


def normalize_dimension_label(value: str) -> str:
    raw = safe_text(value).lower()
    if raw in {"policy_dimension", "policy"}:
        return "Policy"
    if raw in {"impact_dimension", "impact"}:
        return "Impact"
    return safe_text(value)


class MainQuestionPipeline:
    def __init__(self, top_k: int = 8):
        self.top_k = top_k

        registry_path = PROCESSED_DIR / "registry_odm_demo_policy_impact.csv"
        fallback_registry_path = PROCESSED_DIR / "registry_odm_selected_questions.csv"

        if registry_path.exists():
            self.registry_df = pd.read_csv(registry_path)
        else:
            self.registry_df = pd.read_csv(fallback_registry_path)

        self.registry_df["question_id"] = self.registry_df["question_id"].astype(str).str.strip()
        self.registry_df["dimension"] = self.registry_df["dimension"].map(normalize_dimension_label)

        self.retrieval_model = get_retrieval_model()
        self.openai_client = get_openai_client()

    def get_question_metadata(self, question_id: str) -> dict:
        qid = safe_text(question_id)
        rows = self.registry_df[self.registry_df["question_id"] == qid]
        if rows.empty:
            return {}

        row = rows.iloc[0]
        return {
            "question_id": qid,
            "dimension": normalize_dimension_label(row.get("dimension", "")),
            "question": safe_text(row.get("question", "")),
            "response_scoring": safe_text(row.get("response_scoring", "")),
        }

    def get_runtime_dir(self, country: str, question_id: str) -> Path:
        country_slug = safe_text(country).lower().replace(" ", "_")
        qid_slug = safe_text(question_id).lower().replace(" ", "_")
        runtime_dir = UI_RUNTIME_DIR / country_slug / qid_slug
        runtime_dir.mkdir(parents=True, exist_ok=True)
        return runtime_dir

    def build_runtime_evidence(
        self,
        country: str,
        question_id: str,
        question_text: str,
        dimension: str,
        progress_callback=None,
    ):
        runtime_dir = self.get_runtime_dir(country, question_id)

        if progress_callback:
            progress_callback(15, "Generating search queries...")

        queries_df = generate_queries_for_question(
            country=country,
            question_id=question_id,
            question_text=question_text,
            dimension=dimension,
            output_dir=runtime_dir,
        )

        if progress_callback:
            progress_callback(30, "Searching the web...")

        search_df = search_web_for_question(
            country=country,
            question_id=question_id,
            queries_df=queries_df,
            output_dir=runtime_dir,
        )

        if progress_callback:
            progress_callback(45, "Downloading documents...")

        downloaded_df = download_docs_for_question(
            country=country,
            question_id=question_id,
            search_df=search_df,
            output_dir=runtime_dir,
        )

        if progress_callback:
            progress_callback(60, "Extracting text...")

        extracted_df = extract_text_for_question(
            country=country,
            question_id=question_id,
            downloaded_df=downloaded_df,
            output_dir=runtime_dir,
        )

        if progress_callback:
            progress_callback(72, "Chunking text...")

        chunk_df = chunk_text_for_question(
            country=country,
            question_id=question_id,
            extracted_df=extracted_df,
            output_dir=runtime_dir,
        )

        if progress_callback:
            progress_callback(84, "Embedding chunks...")

        runtime_collection = build_runtime_chroma_collection(
            chunk_df=chunk_df,
            model=self.retrieval_model,
        )

        return runtime_dir, runtime_collection

    def run_question(
        self,
        country: str,
        question_id: str,
        progress_callback=None,
    ) -> dict:
        country = safe_text(country)
        question_id = safe_text(question_id)

        if progress_callback:
            progress_callback(5, "Loading question metadata...")

        meta = self.get_question_metadata(question_id)
        if not meta:
            if progress_callback:
                progress_callback(100, "Unsupported question.")
            return {
                "mode": "unsupported",
                "country": country,
                "question_id": question_id,
                "dimension": "",
                "question": "",
                "response_scoring": {},
                "final_answer_selected": "",
                "score_suggestion": None,
                "justification_generated": "Question metadata not found.",
                "confidence": None,
                "evidence_sufficiency": "weak",
                "retrieved_sources": [],
            }

        question_text = meta["question"]
        dimension = meta["dimension"]
        response_scoring = meta["response_scoring"]

        # Every live run scrapes and retrieves fresh evidence. The persistent
        # evaluation collection is intentionally never read here: the deployed
        # demo should reflect the current state of the web, not the evidence
        # snapshot the benchmark results were generated against.
        runtime_dir, runtime_collection = self.build_runtime_evidence(
            country=country,
            question_id=question_id,
            question_text=question_text,
            dimension=dimension,
            progress_callback=progress_callback,
        )

        if progress_callback:
            progress_callback(92, "Retrieving top-k evidence...")

        if runtime_collection is None:
            retrieved_df = pd.DataFrame()
        else:
            retrieved_df = retrieve_topk_for_question_chroma(
                country=country,
                question_id=question_id,
                question_text=question_text,
                dimension=dimension,
                top_k=self.top_k,
                model=self.retrieval_model,
                collection=runtime_collection,
                initial_candidates=runtime_collection.count(),
            )
        mode = "live_main_pipeline"

        if progress_callback:
            progress_callback(96, "Generating justification...")

        result = generate_answer_for_question(
            country=country,
            question_id=question_id,
            question_text=question_text,
            dimension=dimension,
            response_scoring=response_scoring,
            retrieved_df=retrieved_df,
            client=self.openai_client,
        )

        canonical_final_answer = canonicalize_final_answer(
            result.get("final_answer_selected", ""),
            response_scoring=result.get("response_scoring_dict", {}),
        )

        if progress_callback:
            progress_callback(100, "Completed.")

        return {
            "mode": mode,
            "country": country,
            "question_id": question_id,
            "dimension": dimension,
            "question": question_text,
            "response_scoring": result.get("response_scoring_dict", {}),
            "final_answer_selected": canonical_final_answer,
            "score_suggestion": result.get("score_suggestion", None),
            "justification_generated": result.get("justification_generated", ""),
            "confidence": result.get("confidence", None),
            "evidence_sufficiency": result.get("evidence_sufficiency", ""),
            "retrieved_sources": retrieved_df.to_dict(orient="records") if not retrieved_df.empty else [],
        }


def run_batch_pipeline(question_df: pd.DataFrame, top_k: int = 8) -> pd.DataFrame:
    pipeline = MainQuestionPipeline(top_k=top_k)
    rows = []

    total = len(question_df)

    for idx, (_, row) in enumerate(question_df.iterrows(), start=1):
        country = safe_text(row.get("country", ""))
        question_id = safe_text(row.get("question_id", ""))

        print(f"[{idx}/{total}] Running main pipeline for {country} | {question_id}")

        result = pipeline.run_question(
            country=country,
            question_id=question_id,
            progress_callback=None,
        )

        rows.append({
            "country": result.get("country", ""),
            "question_id": result.get("question_id", ""),
            "dimension": result.get("dimension", ""),
            "question": result.get("question", ""),
            "response_scoring": result.get("response_scoring", {}),
            "final_answer_selected": result.get("final_answer_selected", ""),
            "score_suggestion": result.get("score_suggestion", None),
            "justification_generated": result.get("justification_generated", ""),
            "confidence": result.get("confidence", None),
            "evidence_sufficiency": result.get("evidence_sufficiency", ""),
            "mode": result.get("mode", ""),
        })

    return pd.DataFrame(rows)