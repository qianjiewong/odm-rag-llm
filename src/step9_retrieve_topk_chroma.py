import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import PROCESSED_DIR, INTERIM_DIR
from utils import print_header, save_csv, ensure_dir


EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
TOP_K = 8
INITIAL_CANDIDATES = 80

CHROMA_DIRNAME = "chroma_db"
CHROMA_COLLECTION_NAME = "odm_chunks_selected"

MAX_CHUNKS_PER_SOURCE_DEFAULT = 2

BLOCKED_URL_PATTERNS = [
    "open-data-maturity",
    "factsheet",
    "country-factsheet",
    "questionnaire",
    "self-assessment",
    "2025_odm_questionnaire",
]

BLOCKED_TITLE_PATTERNS = [
    "open data maturity",
    "factsheet",
    "questionnaire",
    "self-assessment",
]


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_query_text(question_id: str, question: str, dimension: str) -> str:
    question_id = safe_text(question_id)
    question = safe_text(question)
    dimension = safe_text(dimension)
    return f"{dimension} | {question_id} | {question}"


def get_domain(url: str) -> str:
    try:
        return urlparse(safe_text(url)).netloc.lower()
    except Exception:
        return ""


def is_blocked_source(url: str, title: str) -> bool:
    url_l = safe_text(url).lower()
    title_l = safe_text(title).lower()

    if any(pattern in url_l for pattern in BLOCKED_URL_PATTERNS):
        return True

    if any(pattern in title_l for pattern in BLOCKED_TITLE_PATTERNS):
        return True

    return False


def get_retrieval_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


def get_chroma_client(chroma_dir):
    return chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )


def get_chroma_collection():
    chroma_dir = INTERIM_DIR / CHROMA_DIRNAME
    if not chroma_dir.exists():
        raise FileNotFoundError(
            f"Chroma DB directory not found: {chroma_dir}\n"
            f"Please run step8_embed_chunks_chroma.py first."
        )

    client = get_chroma_client(chroma_dir)
    collection = client.get_collection(CHROMA_COLLECTION_NAME)
    return collection


def query_chroma_candidates(
    collection,
    query_embedding: np.ndarray,
    country: str,
    n_results: int = INITIAL_CANDIDATES,
):
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        where={"country": country},
        include=["documents", "metadatas", "distances"],
    )
    return results


def chroma_distance_to_similarity(distance_value) -> float:
    try:
        return float(-float(distance_value))
    except Exception:
        return -999.0


def normalize_source_key(title: str, url: str) -> str:
    title_key = safe_text(title).lower()
    url_key = safe_text(url).lower()
    return f"{title_key}||{url_key}"


def is_official_domain(domain: str) -> bool:
    d = safe_text(domain).lower()
    official_markers = [
        ".gov", ".gouv", ".gv", ".go.", ".admin", ".parliament", ".senat",
        "government", "gov.", "minister", "ministry", "parliament",
        "europa.eu", "data.gov", "administration", "bund.", "gob.", "data.",
    ]
    return any(marker in d for marker in official_markers)


def get_official_source_bonus(source_url: str, source_title: str, source_file_type: str) -> float:
    domain = get_domain(source_url)
    title_l = safe_text(source_title).lower()
    file_type_l = safe_text(source_file_type).lower()

    bonus = 0.0

    if is_official_domain(domain):
        bonus += 0.08

    if any(x in title_l for x in ["ministry", "government", "strategy", "plan", "law", "decree", "directive", "report"]):
        bonus += 0.04

    if "pdf" in file_type_l:
        bonus += 0.02

    return bonus


def get_low_value_penalty(source_title: str, source_url: str, chunk_text: str) -> float:
    title_l = safe_text(source_title).lower()
    url_l = safe_text(source_url).lower()
    chunk_l = safe_text(chunk_text).lower()

    combined = " ".join([title_l, url_l, chunk_l])

    penalty = 0.0

    generic_markers = [
        "home page",
        "homepage",
        "landing page",
        "welcome",
        "portal",
        "open government data",
        "digital austria",
    ]

    if sum(marker in combined for marker in generic_markers) >= 2:
        penalty += 0.05

    if len(chunk_l.split()) < 40:
        penalty += 0.03

    return penalty


def get_question_rerank_bonus(
    question_id: str,
    dimension: str,
    source_title: str,
    source_url: str,
    chunk_text: str,
    source_file_type: str,
) -> float:
    qid = safe_text(question_id)
    dim = safe_text(dimension).lower()

    title_l = safe_text(source_title).lower()
    url_l = safe_text(source_url).lower()
    chunk_l = safe_text(chunk_text).lower()
    file_type_l = safe_text(source_file_type).lower()

    combined = " ".join([title_l, url_l, chunk_l, file_type_l])

    bonus = 0.0

    def has_any(words):
        return any(w in combined for w in words)

    if qid in {"P1", "P2", "P4"}:
        if has_any(["law", "act", "directive", "decree", "strategy", "roadmap", "plan", "policy"]):
            bonus += 0.16
        if has_any(["ministry", "government", "official", "administration"]):
            bonus += 0.05

    if qid in {"P13", "P22", "P24"}:
        if has_any(["governance", "coordination", "steering", "publication", "plan", "review", "update", "guideline"]):
            bonus += 0.14

    if qid == "P3":
        if has_any(["regional", "local", "municipal", "city", "province", "county", "local authority"]):
            bonus += 0.18

    if qid == "P16":
        if has_any(["public bodies", "authorities", "municipalities", "agencies", "ministries", "regions", "local authorities"]):
            bonus += 0.16

    if qid == "I1":
        if has_any(["definition", "defined", "means", "reuse", "re-use", "methodology", "framework", "glossary"]):
            bonus += 0.18
        if has_any(["law", "directive", "act"]):
            bonus += 0.06

    if qid in {"I2", "I10"}:
        if has_any(["monitor", "monitoring", "dashboard", "report", "case", "reuse", "showcase", "catalogue", "portal", "methodology"]):
            bonus += 0.16

    if qid == "I9":
        if has_any(["consultation", "survey", "workshop", "engagement", "stakeholder", "community", "user needs"]):
            bonus += 0.16

    if qid in {"I12", "I17", "I27"}:
        if has_any(["impact", "study", "report", "evaluation", "economic", "social", "government", "transparency", "innovation", "benefit"]):
            bonus += 0.16

    if dim == "policy" and has_any(["strategy", "policy", "law", "decree", "directive", "plan"]):
        bonus += 0.03

    if dim == "impact" and has_any(["impact", "report", "study", "evaluation"]):
        bonus += 0.03

    return bonus


def get_belgium_targeted_bonus(
    country: str,
    question_id: str,
    source_title: str,
    source_url: str,
    chunk_text: str,
    source_file_type: str,
) -> float:
    """
    Belgium-only targeted refinement for weak questions:
    I10, P24, P2, I2, I1
    """
    if safe_text(country).lower() != "belgium":
        return 0.0

    qid = safe_text(question_id)
    title_l = safe_text(source_title).lower()
    url_l = safe_text(source_url).lower()
    chunk_l = safe_text(chunk_text).lower()
    file_type_l = safe_text(source_file_type).lower()

    combined = " ".join([title_l, url_l, chunk_l, file_type_l])

    bonus = 0.0

    def has_any(words):
        return any(w in combined for w in words)

    # I10: ad-hoc vs systematic reuse collection/showcase
    if qid == "I10":
        if has_any([
            "reuse case", "re-use case", "showcase", "examples of reuse", "case study",
            "catalogue", "inventory", "ad hoc", "ad-hoc", "sporadic"
        ]):
            bonus += 0.18
        if has_any(["report", "portal", "monitoring", "collection of examples"]):
            bonus += 0.06

    # P24: policy update / review / approval status
    if qid == "P24":
        if has_any([
            "update", "review", "revision", "revised", "approval", "approved",
            "not yet approved", "federal level", "adoption", "amendment"
        ]):
            bonus += 0.22
        if has_any(["policy", "strategy", "ordinance", "legal framework"]):
            bonus += 0.05

    # P2: national strategy / plan / official strategy document
    if qid == "P2":
        if has_any([
            "strategy", "open data strategy", "federal strategy", "national strategy",
            "plan", "roadmap", "policy document", "official document"
        ]):
            bonus += 0.18
        if has_any(["pdf", "report", "document"]):
            bonus += 0.05

    # I2: monitoring / methodology / support process
    if qid == "I2":
        if has_any([
            "monitor", "monitoring", "methodology", "dashboard", "report",
            "tracking", "support", "process", "measurement", "evaluation"
        ]):
            bonus += 0.18
        if has_any(["portal", "annual report", "activity report"]):
            bonus += 0.05

    # I1: compact definition / legal wording of reuse
    if qid == "I1":
        if has_any([
            "definition", "shall mean", "means", "reuse", "re-use",
            "public sector information", "for the purposes of", "article"
        ]):
            bonus += 0.20
        if has_any(["law", "ordinance", "directive", "legal"]):
            bonus += 0.06

    return bonus


def get_belgium_targeted_penalty(
    country: str,
    question_id: str,
    source_title: str,
    source_url: str,
    chunk_text: str,
) -> float:
    """
    Belgium-only penalty for overly generic or over-explanatory evidence
    on weak questions.
    """
    if safe_text(country).lower() != "belgium":
        return 0.0

    qid = safe_text(question_id)
    title_l = safe_text(source_title).lower()
    url_l = safe_text(source_url).lower()
    chunk_l = safe_text(chunk_text).lower()

    combined = " ".join([title_l, url_l, chunk_l])

    penalty = 0.0

    def has_any(words):
        return any(w in combined for w in words)

    # P24: penalize generic legal/policy pages that do not mention update/review status
    if qid == "P24":
        if has_any(["legal framework", "policy framework", "strategy framework"]) and not has_any(
            ["update", "review", "revision", "approved", "not yet approved", "adoption"]
        ):
            penalty += 0.10

    # P2: penalize broad portal text if no actual strategy-doc signal
    if qid == "P2":
        if has_any(["portal", "homepage", "open data portal"]) and not has_any(
            ["strategy", "roadmap", "policy document", "plan", "pdf"]
        ):
            penalty += 0.10

    # I2: penalize generic support text if no monitoring/methodology signal
    if qid == "I2":
        if has_any(["support", "open data help", "portal information"]) and not has_any(
            ["monitor", "monitoring", "methodology", "tracking", "evaluation", "report"]
        ):
            penalty += 0.10

    # I10: penalize long generic reuse language if no case/showcase/ad-hoc signal
    if qid == "I10":
        if has_any(["reuse", "re-use"]) and not has_any(
            ["case", "showcase", "example", "ad hoc", "ad-hoc", "sporadic", "catalogue"]
        ):
            penalty += 0.10

    # I1: penalize promotional text if no explicit definition/legal wording
    if qid == "I1":
        if has_any(["benefits of open data", "value of open data", "promoting open data"]):
            penalty += 0.10
        if not has_any(["definition", "means", "reuse", "re-use", "article", "public sector information"]):
            penalty += 0.05

    return penalty


def get_dynamic_max_chunks_per_source(question_id: str) -> int:
    qid = safe_text(question_id)

    if qid in {"I1", "I2", "I10", "I12", "I17", "I27", "P1", "P2", "P4", "P24"}:
        return 1

    return MAX_CHUNKS_PER_SOURCE_DEFAULT


def diversify_candidates(
    candidate_rows: list[dict],
    question_id: str,
    top_k: int = TOP_K,
) -> list[dict]:
    max_chunks_per_source = get_dynamic_max_chunks_per_source(question_id)

    selected = []
    selected_ids = set()
    source_counts = {}

    # Pass 1: one per source
    for row in candidate_rows:
        source_key = normalize_source_key(
            row.get("source_title", ""),
            row.get("source_url", ""),
        )

        if source_counts.get(source_key, 0) >= 1:
            continue

        row_id = f"{row.get('chunk_id', '')}__{row.get('source_url', '')}"
        if row_id in selected_ids:
            continue

        selected.append(row)
        selected_ids.add(row_id)
        source_counts[source_key] = 1

        if len(selected) >= top_k:
            return selected

    # Pass 2: backfill
    for row in candidate_rows:
        source_key = normalize_source_key(
            row.get("source_title", ""),
            row.get("source_url", ""),
        )

        row_id = f"{row.get('chunk_id', '')}__{row.get('source_url', '')}"
        if row_id in selected_ids:
            continue

        if source_counts.get(source_key, 0) >= max_chunks_per_source:
            continue

        selected.append(row)
        selected_ids.add(row_id)
        source_counts[source_key] = source_counts.get(source_key, 0) + 1

        if len(selected) >= top_k:
            return selected

    return selected[:top_k]


def retrieve_topk_for_question_chroma(
    country: str,
    question_id: str,
    question_text: str,
    dimension: str,
    top_k: int = TOP_K,
    model=None,
    collection=None,
    initial_candidates: int = INITIAL_CANDIDATES,
) -> pd.DataFrame:
    country = safe_text(country)
    question_id = safe_text(question_id)
    question_text = safe_text(question_text)
    dimension = safe_text(dimension)

    if model is None:
        model = get_retrieval_model()

    if collection is None:
        collection = get_chroma_collection()

    query_text = build_query_text(question_id, question_text, dimension)

    query_embedding = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")[0]

    raw_results = query_chroma_candidates(
        collection=collection,
        query_embedding=query_embedding,
        country=country,
        n_results=initial_candidates,
    )

    metadatas = raw_results.get("metadatas", [[]])[0]
    documents = raw_results.get("documents", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]

    if not metadatas:
        return pd.DataFrame()

    candidates = []

    for metadata, document, distance in zip(metadatas, documents, distances):
        source_title = safe_text(metadata.get("title", ""))
        source_url = safe_text(metadata.get("url", ""))
        source_file_type = safe_text(metadata.get("file_type", ""))
        chunk_text = safe_text(document)
        source_domain = get_domain(source_url)

        if is_blocked_source(source_url, source_title):
            continue

        base_similarity = chroma_distance_to_similarity(distance)

        official_bonus = get_official_source_bonus(
            source_url=source_url,
            source_title=source_title,
            source_file_type=source_file_type,
        )

        rerank_bonus = get_question_rerank_bonus(
            question_id=question_id,
            dimension=dimension,
            source_title=source_title,
            source_url=source_url,
            chunk_text=chunk_text,
            source_file_type=source_file_type,
        )

        low_value_penalty = get_low_value_penalty(
            source_title=source_title,
            source_url=source_url,
            chunk_text=chunk_text,
        )

        belgium_bonus = get_belgium_targeted_bonus(
            country=country,
            question_id=question_id,
            source_title=source_title,
            source_url=source_url,
            chunk_text=chunk_text,
            source_file_type=source_file_type,
        )

        belgium_penalty = get_belgium_targeted_penalty(
            country=country,
            question_id=question_id,
            source_title=source_title,
            source_url=source_url,
            chunk_text=chunk_text,
        )

        final_score = (
            base_similarity
            + official_bonus
            + rerank_bonus
            + belgium_bonus
            - low_value_penalty
            - belgium_penalty
        )

        candidates.append({
            "country": country,
            "question_id": question_id,
            "dimension": dimension,
            "question": question_text,
            "retrieval_query": query_text,

            "source_question_id": safe_text(metadata.get("question_id", "")),
            "source_title": source_title,
            "source_url": source_url,
            "source_domain": source_domain,
            "source_saved_path": safe_text(metadata.get("saved_path", "")),
            "source_file_type": source_file_type,
            "chunk_id": safe_text(metadata.get("chunk_id", "")),
            "chunk_index": metadata.get("chunk_index", ""),
            "chunk_text": chunk_text,
            "chunk_length_words": metadata.get("chunk_length_words", ""),

            "similarity_score": base_similarity,
            "official_bonus": official_bonus,
            "rerank_bonus": rerank_bonus,
            "belgium_bonus": belgium_bonus,
            "low_value_penalty": low_value_penalty,
            "belgium_penalty": belgium_penalty,
            "final_retrieval_score": final_score,
        })

    if not candidates:
        return pd.DataFrame()

    candidates = sorted(
        candidates,
        key=lambda x: x["final_retrieval_score"],
        reverse=True,
    )

    selected = diversify_candidates(
        candidate_rows=candidates,
        question_id=question_id,
        top_k=top_k,
    )

    if not selected:
        return pd.DataFrame()

    for rank, row in enumerate(selected, start=1):
        row["retrieval_rank"] = rank

    retrieved_df = pd.DataFrame(selected)

    ordered_cols = [
        "country",
        "question_id",
        "dimension",
        "question",
        "retrieval_query",
        "retrieval_rank",
        "similarity_score",
        "official_bonus",
        "rerank_bonus",
        "belgium_bonus",
        "low_value_penalty",
        "belgium_penalty",
        "final_retrieval_score",
        "source_question_id",
        "source_title",
        "source_url",
        "source_domain",
        "source_saved_path",
        "source_file_type",
        "chunk_id",
        "chunk_index",
        "chunk_text",
        "chunk_length_words",
    ]

    existing_cols = [c for c in ordered_cols if c in retrieved_df.columns]
    remaining_cols = [c for c in retrieved_df.columns if c not in existing_cols]

    return retrieved_df[existing_cols + remaining_cols]


def main():
    print_header("STEP 9 EXPERIMENT: RETRIEVE TOP-K CHUNKS WITH CHROMADB (BELGIUM TARGETED)")

    ground_truth_path = PROCESSED_DIR / "ground_truth_odm_selected.csv"
    output_dir = INTERIM_DIR / "retrieval"
    output_path = output_dir / "retrieved_topk_odm_selected_chroma.csv"

    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {ground_truth_path}\n"
            f"Please run step2_build_ground_truth.py first."
        )

    ensure_dir(output_dir)

    gt_df = pd.read_csv(ground_truth_path)

    print(f"Loading retrieval model: {EMBED_MODEL_NAME}")
    model = get_retrieval_model()

    print("Loading Chroma collection...")
    collection = get_chroma_collection()

    rows = []

    for _, row in gt_df.iterrows():
        country = safe_text(row.get("country", ""))
        question_id = safe_text(row.get("question_id", ""))
        question = safe_text(row.get("question", ""))
        dimension = safe_text(row.get("dimension", ""))

        retrieved_df = retrieve_topk_for_question_chroma(
            country=country,
            question_id=question_id,
            question_text=question,
            dimension=dimension,
            top_k=TOP_K,
            model=model,
            collection=collection,
            initial_candidates=INITIAL_CANDIDATES,
        )

        if retrieved_df.empty:
            print(f"No Belgium-targeted Chroma retrieval output for {country} | {question_id}")
            continue

        rows.extend(retrieved_df.to_dict(orient="records"))

        unique_sources = retrieved_df[["source_title", "source_url"]].drop_duplicates().shape[0]
        print(
            f"Retrieved {len(retrieved_df)} Belgium-targeted Chroma chunks for "
            f"{country} | {question_id} | unique_sources={unique_sources}"
        )

    out_df = pd.DataFrame(rows)
    save_csv(out_df, output_path)

    print(f"\nSaved Belgium-targeted Chroma retrieval file to: {output_path}")
    print(f"Rows: {len(out_df)}")

    if len(out_df) > 0:
        preview_cols = [
            "country",
            "question_id",
            "retrieval_rank",
            "similarity_score",
            "official_bonus",
            "rerank_bonus",
            "belgium_bonus",
            "low_value_penalty",
            "belgium_penalty",
            "final_retrieval_score",
            "source_title",
        ]
        preview_cols = [c for c in preview_cols if c in out_df.columns]

        print("\nPreview:")
        print(out_df[preview_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()