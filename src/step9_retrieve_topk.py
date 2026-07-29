import numpy as np
import pandas as pd
import pickle

from sentence_transformers import SentenceTransformer

from config import PROCESSED_DIR, INTERIM_DIR
from utils import print_header, save_csv, ensure_dir


EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
TOP_K = 8
MAX_CHUNKS_PER_SOURCE = 2

# Step 9 safety blocking only
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

# Soft reranking bonuses
QUESTION_HINT_BONUS = 0.08
I1_LEGAL_SOURCE_EXTRA_BONUS = 0.12


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_query_text(question_id: str, question: str, dimension: str) -> str:
    question_id = safe_text(question_id)
    question = safe_text(question)
    dimension = safe_text(dimension)
    return f"{dimension} | {question_id} | {question}"


def cosine_similarity_matrix(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.dot(matrix, query_vector)


def is_blocked_source(url: str, title: str) -> bool:
    url_l = safe_text(url).lower()
    title_l = safe_text(title).lower()

    if any(pattern in url_l for pattern in BLOCKED_URL_PATTERNS):
        return True

    if any(pattern in title_l for pattern in BLOCKED_TITLE_PATTERNS):
        return True

    return False


def load_retrieval_assets():
    embeddings_path = INTERIM_DIR / "embeddings" / "chunks_odm_selected.npy"
    metadata_path = INTERIM_DIR / "embeddings" / "chunks_odm_selected_metadata.pkl"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {embeddings_path}\n"
            f"Please run step8_embed_chunks.py first."
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            f"Please run step8_embed_chunks.py first."
        )

    chunk_embeddings = np.load(embeddings_path)

    with open(metadata_path, "rb") as f:
        chunk_df = pickle.load(f)

    if len(chunk_df) != len(chunk_embeddings):
        raise ValueError(
            f"Mismatch between metadata rows ({len(chunk_df)}) and embeddings rows ({len(chunk_embeddings)})."
        )

    return chunk_df, chunk_embeddings


def get_retrieval_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


def build_source_key(row: pd.Series) -> str:
    url = safe_text(row.get("url", ""))
    title = safe_text(row.get("title", ""))
    if url:
        return url.lower()
    if title:
        return title.lower()
    return "unknown_source"


def get_question_rerank_keywords(question_id: str) -> dict:
    """
    Returns question-specific positive keywords for reranking.
    These are soft hints, not hard rules.
    """
    qid = safe_text(question_id)

    keyword_map = {
        "I1": {
            "title_or_url": [
                "law", "act", "legal", "definition", "public information",
                "public sector information", "reuse", "taaskasutus",
                "avaliku teabe", "riigiteataja"
            ],
            "chunk_text": [
                "definition", "defined", "law", "act", "reuse of public information",
                "public sector information", "public information", "taaskasutus",
                "avaliku teabe"
            ],
        },
        "I2": {
            "title_or_url": [
                "monitoring", "methodology", "dashboard", "assessment",
                "indikator", "indicator", "annual", "report", "metoodika",
                "measurement"
            ],
            "chunk_text": [
                "monitoring", "methodology", "dashboard", "indicator",
                "indicators", "assessment", "annual assessment",
                "measurement", "metrics", "report"
            ],
        },
        "I10": {
            "title_or_url": [
                "case", "cases", "example", "examples", "showcase",
                "reuse case", "reuse cases", "case study"
            ],
            "chunk_text": [
                "example", "examples", "showcase", "case study",
                "reuse case", "reuse cases", "used by", "application of open data"
            ],
        },
        "I12": {
            "title_or_url": [
                "impact", "assessment", "report", "efficiency",
                "service improvement", "transparency", "government impact"
            ],
            "chunk_text": [
                "impact", "efficiency", "transparency", "improvement",
                "better services", "government impact", "administrative burden"
            ],
        },
        "I17": {
            "title_or_url": [
                "social impact", "education", "inclusion", "awareness", "public value"
            ],
            "chunk_text": [
                "social impact", "education", "inclusion", "awareness",
                "public value", "citizens", "society"
            ],
        },
        "I27": {
            "title_or_url": [
                "economic impact", "innovation", "business", "startup",
                "value creation", "market", "economic"
            ],
            "chunk_text": [
                "economic impact", "innovation", "business", "startup",
                "value creation", "market", "economic", "entrepreneur"
            ],
        },
    }

    return keyword_map.get(qid, {"title_or_url": [], "chunk_text": []})


def compute_question_rerank_bonus(question_id: str, row: pd.Series) -> float:
    """
    Computes a small bonus if the source/chunk text looks like the right evidence type
    for the specific question.
    """
    hints = get_question_rerank_keywords(question_id)
    title_or_url_keywords = hints.get("title_or_url", [])
    chunk_keywords = hints.get("chunk_text", [])

    title = safe_text(row.get("title", "")).lower()
    url = safe_text(row.get("url", "")).lower()
    chunk_text = safe_text(row.get("chunk_text", "")).lower()

    score = 0.0

    for kw in title_or_url_keywords:
        kw_l = kw.lower()
        if kw_l in title or kw_l in url:
            score += 0.04

    for kw in chunk_keywords:
        kw_l = kw.lower()
        if kw_l in chunk_text:
            score += 0.02

    return min(score, QUESTION_HINT_BONUS)


def compute_i1_legal_source_bonus(question_id: str, row: pd.Series) -> float:
    """
    Extra legal-source preference for I1 only.
    This is intentionally stronger than the general rerank bonus because I1
    needs a formal legal definition source.
    """
    if safe_text(question_id) != "I1":
        return 0.0

    title = safe_text(row.get("title", "")).lower()
    url = safe_text(row.get("url", "")).lower()
    chunk_text = safe_text(row.get("chunk_text", "")).lower()

    bonus = 0.0

    strong_legal_title_or_url_terms = [
        "riigiteataja",
        "law",
        "act",
        "legal",
        "statute",
        "public information act",
        "avaliku teabe seadus",
    ]

    strong_legal_chunk_terms = [
        "definition",
        "defined as",
        "means",
        "public information act",
        "avaliku teabe seadus",
        "reuse of public information",
        "public sector information",
        "taaskasutus",
    ]

    for kw in strong_legal_title_or_url_terms:
        if kw in title or kw in url:
            bonus += 0.05

    for kw in strong_legal_chunk_terms:
        if kw in chunk_text:
            bonus += 0.03

    return min(bonus, I1_LEGAL_SOURCE_EXTRA_BONUS)


def diversify_ranked_results(
    candidate_df: pd.DataFrame,
    base_sims: np.ndarray,
    adjusted_sims: np.ndarray,
    top_k: int,
    max_chunks_per_source: int = MAX_CHUNKS_PER_SOURCE,
) -> list[tuple[int, float, float]]:
    """
    Returns a list of (local_index, base_similarity_score, adjusted_similarity_score)
    after diversification.
    """
    ranked_local_indices = np.argsort(-adjusted_sims)
    selected = []
    source_counts = {}

    for local_idx in ranked_local_indices:
        row = candidate_df.iloc[local_idx]
        source_key = build_source_key(row)

        current_count = source_counts.get(source_key, 0)
        if current_count >= max_chunks_per_source:
            continue

        selected.append((
            local_idx,
            float(base_sims[local_idx]),
            float(adjusted_sims[local_idx]),
        ))
        source_counts[source_key] = current_count + 1

        if len(selected) >= top_k:
            break

    return selected


def retrieve_topk_for_question(
    country: str,
    question_id: str,
    question_text: str,
    dimension: str,
    top_k: int = TOP_K,
    model=None,
    chunk_df=None,
    chunk_embeddings=None,
) -> pd.DataFrame:
    """
    Reusable single-question retrieval function.
    This is what pipeline_runner.py and the UI should call.
    """

    country = safe_text(country)
    question_id = safe_text(question_id)
    question_text = safe_text(question_text)
    dimension = safe_text(dimension)

    if chunk_df is None or chunk_embeddings is None:
        chunk_df, chunk_embeddings = load_retrieval_assets()

    if model is None:
        model = get_retrieval_model()

    query_text = build_query_text(question_id, question_text, dimension)

    query_embedding = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")[0]

    # Filter to same country only
    country_mask = chunk_df["country"].astype(str).str.strip() == country
    country_chunk_df = chunk_df[country_mask].copy()

    if country_chunk_df.empty:
        return pd.DataFrame()

    # Block suspicious answer-leakage sources
    blocked_mask = country_chunk_df.apply(
        lambda r: is_blocked_source(
            r.get("url", ""),
            r.get("title", "")
        ),
        axis=1
    )
    country_chunk_df = country_chunk_df[~blocked_mask].copy()

    if country_chunk_df.empty:
        return pd.DataFrame()

    # Retrieve only from filtered country chunk pool
    country_indices = country_chunk_df.index.to_numpy()
    country_embeddings = chunk_embeddings[country_indices]

    base_sims = cosine_similarity_matrix(query_embedding, country_embeddings)

    candidate_reset_df = country_chunk_df.reset_index(drop=True)

    general_bonuses = np.array([
        compute_question_rerank_bonus(question_id, candidate_reset_df.iloc[i])
        for i in range(len(candidate_reset_df))
    ], dtype="float32")

    i1_legal_bonuses = np.array([
        compute_i1_legal_source_bonus(question_id, candidate_reset_df.iloc[i])
        for i in range(len(candidate_reset_df))
    ], dtype="float32")

    total_bonuses = general_bonuses + i1_legal_bonuses
    adjusted_sims = base_sims + total_bonuses

    diversified = diversify_ranked_results(
        candidate_df=candidate_reset_df,
        base_sims=base_sims,
        adjusted_sims=adjusted_sims,
        top_k=top_k,
        max_chunks_per_source=MAX_CHUNKS_PER_SOURCE,
    )

    rows = []

    for rank, (local_idx, base_sim_score, adjusted_sim_score) in enumerate(diversified, start=1):
        global_idx = country_indices[local_idx]
        chunk_row = chunk_df.iloc[global_idx]

        rows.append({
            "country": country,
            "question_id": question_id,
            "dimension": dimension,
            "question": question_text,
            "retrieval_query": query_text,
            "retrieval_rank": rank,
            "similarity_score": base_sim_score,
            "adjusted_similarity_score": adjusted_sim_score,
            "rerank_bonus": adjusted_sim_score - base_sim_score,

            "source_question_id": safe_text(chunk_row.get("question_id", "")),
            "source_title": safe_text(chunk_row.get("title", "")),
            "source_url": safe_text(chunk_row.get("url", "")),
            "source_saved_path": safe_text(chunk_row.get("saved_path", "")),
            "source_file_type": safe_text(chunk_row.get("file_type", "")),
            "chunk_id": safe_text(chunk_row.get("chunk_id", "")),
            "chunk_index": chunk_row.get("chunk_index", ""),
            "chunk_text": safe_text(chunk_row.get("chunk_text", "")),
            "chunk_length_words": chunk_row.get("chunk_length_words", ""),
        })

    return pd.DataFrame(rows)


def main():
    print_header("STEP 9: RETRIEVE TOP-K CHUNKS")

    ground_truth_path = PROCESSED_DIR / "ground_truth_odm_selected.csv"
    output_dir = INTERIM_DIR / "retrieval"
    output_path = output_dir / "retrieved_topk_odm_selected.csv"

    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {ground_truth_path}\n"
            f"Please run step2_build_ground_truth.py first."
        )

    ensure_dir(output_dir)

    gt_df = pd.read_csv(ground_truth_path)

    print(f"Loading retrieval model: {EMBED_MODEL_NAME}")
    model = get_retrieval_model()
    chunk_df, chunk_embeddings = load_retrieval_assets()

    rows = []

    for _, row in gt_df.iterrows():
        country = safe_text(row.get("country", ""))
        question_id = safe_text(row.get("question_id", ""))
        question = safe_text(row.get("question", ""))
        dimension = safe_text(row.get("dimension", ""))

        retrieved_df = retrieve_topk_for_question(
            country=country,
            question_id=question_id,
            question_text=question,
            dimension=dimension,
            top_k=TOP_K,
            model=model,
            chunk_df=chunk_df,
            chunk_embeddings=chunk_embeddings,
        )

        if retrieved_df.empty:
            print(f"No retrieval output for {country} | {question_id}")
            continue

        rows.extend(retrieved_df.to_dict(orient="records"))

        print(
            f"Retrieved {len(retrieved_df)} chunks for {country} | {question_id} "
            f"with diversification + rerank bonus"
        )

    out_df = pd.DataFrame(rows)
    save_csv(out_df, output_path)

    print(f"\nSaved retrieval file to: {output_path}")
    print(f"Rows: {len(out_df)}")

    if len(out_df) > 0:
        preview_df = out_df[[
            "country", "question_id", "retrieval_rank",
            "similarity_score", "adjusted_similarity_score", "rerank_bonus", "source_title"
        ]].head(20)
        print("\nPreview:")
        print(preview_df.to_string(index=False))


if __name__ == "__main__":
    main()