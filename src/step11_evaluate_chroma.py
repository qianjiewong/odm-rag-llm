import re
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from config import PROCESSED_DIR, GENERATION_DIR, EVALUATION_DIR
from utils import print_header, save_csv, ensure_dir


MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SBERT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_text(text: str) -> str:
    text = safe_text(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def canonicalize_answer_label(text: str) -> str:
    raw = safe_text(text)
    if not raw:
        return ""

    normalized = normalize_text(raw)

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
        "approximately half of the public bodies": "Approximately half of the public bodies",
        "few public bodies": "Few public bodies",
    }

    if normalized in canonical_map:
        return canonical_map[normalized]

    return raw


def strip_sources_section(text: str) -> str:
    """
    Remove the trailing ## Sources block from generated justifications
    for similarity evaluation only.
    """
    raw = safe_text(text)
    if not raw:
        return ""

    parts = re.split(r"(?i)##\s*Sources", raw, maxsplit=1)
    return safe_text(parts[0])


def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    return re.findall(r"\b\w+\b", text)


def safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def cosine_from_vectors(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def tfidf_cosine_similarity(text_a: str, text_b: str):
    text_a = safe_text(text_a)
    text_b = safe_text(text_b)

    if not text_a or not text_b:
        return np.nan

    try:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([text_a, text_b])
        sim = sklearn_cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return float(sim)
    except Exception:
        return np.nan


def jaccard_similarity(text_a: str, text_b: str):
    set_a = set(tokenize(text_a))
    set_b = set(tokenize(text_b))

    if not set_a or not set_b:
        return np.nan

    union = len(set_a.union(set_b))
    if union == 0:
        return np.nan

    intersection = len(set_a.intersection(set_b))
    return float(intersection / union)


def bleu_score(text_a: str, text_b: str):
    ref_tokens = tokenize(text_a)
    cand_tokens = tokenize(text_b)

    if not ref_tokens or not cand_tokens:
        return np.nan

    try:
        smoother = SmoothingFunction().method1
        return float(sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoother))
    except Exception:
        return np.nan


def rouge_l_f1(text_a: str, text_b: str):
    text_a = safe_text(text_a)
    text_b = safe_text(text_b)

    if not text_a or not text_b:
        return np.nan

    try:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(text_a, text_b)
        return float(scores["rougeL"].fmeasure)
    except Exception:
        return np.nan


def summarize_scope(df: pd.DataFrame, scope_name: str) -> dict:
    return {
        "scope": scope_name,
        "num_questions": len(df),
        "avg_response_match": df["response_match"].mean(),
        "avg_response_match_normalized": df["response_match_normalized"].mean(),
        "avg_score_difference": df["score_difference"].dropna().mean() if not df["score_difference"].dropna().empty else None,
        "avg_score_abs_difference": df["score_abs_difference"].dropna().mean() if not df["score_abs_difference"].dropna().empty else None,
        "avg_cosine_similarity_minilm": df["cosine_similarity_minilm"].dropna().mean() if not df["cosine_similarity_minilm"].dropna().empty else None,
        "avg_cosine_similarity_sbert": df["cosine_similarity_sbert"].dropna().mean() if not df["cosine_similarity_sbert"].dropna().empty else None,
        "avg_tfidf_similarity": df["tfidf_similarity"].dropna().mean() if not df["tfidf_similarity"].dropna().empty else None,
        "avg_jaccard_similarity": df["jaccard_similarity"].dropna().mean() if not df["jaccard_similarity"].dropna().empty else None,
        "avg_bleu_score": df["bleu_score"].dropna().mean() if not df["bleu_score"].dropna().empty else None,
        "avg_rouge_l_f1": df["rouge_l_f1"].dropna().mean() if not df["rouge_l_f1"].dropna().empty else None,
        "avg_confidence": df["confidence"].dropna().mean() if not df["confidence"].dropna().empty else None,
    }


def print_summary_block(summary: dict, title: str):
    print(f"\n{title}")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


def main():
    print_header("STEP 11 EXPERIMENT: EVALUATE CHROMA ODM OUTPUTS (BODY ONLY)")

    ground_truth_path = PROCESSED_DIR / "ground_truth_odm_selected.csv"
    generated_path = GENERATION_DIR / "generated_answers_odm_selected_chroma.csv"

    output_dir = EVALUATION_DIR
    output_path = output_dir / "evaluation_odm_selected_chroma.csv"
    summary_output_path = output_dir / "evaluation_summary_odm_selected_chroma.csv"
    country_summary_output_path = output_dir / "evaluation_summary_by_country_odm_selected_chroma.csv"

    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {ground_truth_path}\n"
            f"Please run step2_build_ground_truth.py first."
        )

    if not generated_path.exists():
        raise FileNotFoundError(
            f"Generated Chroma answers file not found: {generated_path}\n"
            f"Please run step10_generate_answers_chroma.py first."
        )

    ensure_dir(output_dir)

    gt_df = pd.read_csv(ground_truth_path)
    gen_df = pd.read_csv(generated_path)

    merged_df = gt_df.merge(
        gen_df,
        on=["country", "question_id", "dimension", "question"],
        how="left",
        suffixes=("_gt", "_gen")
    )

    print(f"Loading MiniLM model: {MINILM_MODEL_NAME}")
    minilm_model = SentenceTransformer(MINILM_MODEL_NAME)

    print(f"Loading S-BERT model: {SBERT_MODEL_NAME}")
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME)

    rows = []

    for _, row in merged_df.iterrows():
        country = safe_text(row.get("country", ""))
        question_id = safe_text(row.get("question_id", ""))
        dimension = safe_text(row.get("dimension", ""))
        question = safe_text(row.get("question", ""))

        ground_truth_answer = safe_text(row.get("final_answer", ""))
        ground_truth_explanation = safe_text(row.get("final_explanation", ""))

        generated_answer = safe_text(row.get("final_answer_selected", ""))
        generated_justification = safe_text(row.get("justification_generated", ""))

        # NEW: body-only text for similarity metrics
        generated_justification_body = strip_sources_section(generated_justification)

        ground_truth_score = safe_float(row.get("awarded_score", None))
        generated_score = safe_float(row.get("score_suggestion", None))

        confidence = safe_float(row.get("confidence", None))
        evidence_sufficiency = safe_text(row.get("evidence_sufficiency", ""))

        ground_truth_answer_canonical = canonicalize_answer_label(ground_truth_answer)
        generated_answer_canonical = canonicalize_answer_label(generated_answer)

        response_match = int(ground_truth_answer_canonical == generated_answer_canonical)
        response_match_normalized = int(
            normalize_text(ground_truth_answer) == normalize_text(generated_answer)
        )

        if ground_truth_score is not None and generated_score is not None:
            score_difference = generated_score - ground_truth_score
            score_abs_difference = abs(score_difference)
        else:
            score_difference = None
            score_abs_difference = None

        # CHANGED: compute similarity using body-only justification
        if ground_truth_explanation and generated_justification_body:
            minilm_emb = minilm_model.encode(
                [ground_truth_explanation, generated_justification_body],
                convert_to_numpy=True,
                normalize_embeddings=False
            )
            cosine_similarity_minilm = cosine_from_vectors(minilm_emb[0], minilm_emb[1])

            sbert_emb = sbert_model.encode(
                [ground_truth_explanation, generated_justification_body],
                convert_to_numpy=True,
                normalize_embeddings=False
            )
            cosine_similarity_sbert = cosine_from_vectors(sbert_emb[0], sbert_emb[1])

            tfidf_similarity = tfidf_cosine_similarity(
                ground_truth_explanation, generated_justification_body
            )
            jaccard = jaccard_similarity(
                ground_truth_explanation, generated_justification_body
            )
            bleu = bleu_score(
                ground_truth_explanation, generated_justification_body
            )
            rouge_l = rouge_l_f1(
                ground_truth_explanation, generated_justification_body
            )
        else:
            cosine_similarity_minilm = np.nan
            cosine_similarity_sbert = np.nan
            tfidf_similarity = np.nan
            jaccard = np.nan
            bleu = np.nan
            rouge_l = np.nan

        rows.append({
            "country": country,
            "question_id": question_id,
            "dimension": dimension,
            "question": question,

            "ground_truth_answer": ground_truth_answer,
            "generated_answer": generated_answer,
            "ground_truth_answer_canonical": ground_truth_answer_canonical,
            "generated_answer_canonical": generated_answer_canonical,
            "response_match": response_match,
            "response_match_normalized": response_match_normalized,

            "ground_truth_score": ground_truth_score,
            "generated_score": generated_score,
            "score_difference": score_difference,
            "score_abs_difference": score_abs_difference,

            "ground_truth_explanation": ground_truth_explanation,
            "generated_justification": generated_justification,
            "generated_justification_body": generated_justification_body,

            "cosine_similarity_minilm": cosine_similarity_minilm,
            "cosine_similarity_sbert": cosine_similarity_sbert,
            "tfidf_similarity": tfidf_similarity,
            "jaccard_similarity": jaccard,
            "bleu_score": bleu,
            "rouge_l_f1": rouge_l,

            "confidence": confidence,
            "evidence_sufficiency": evidence_sufficiency,
        })

        print(
            f"Evaluated Chroma {country} | {question_id} | "
            f"match={response_match} | "
            f"norm_match={response_match_normalized} | "
            f"abs_score_diff={score_abs_difference} | "
            f"MiniLM={cosine_similarity_minilm} | "
            f"S-BERT={cosine_similarity_sbert}"
        )

    out_df = pd.DataFrame(rows)
    save_csv(out_df, output_path)

    overall_summary = summarize_scope(out_df, "overall")
    overall_summary_df = pd.DataFrame([overall_summary])
    save_csv(overall_summary_df, summary_output_path)

    country_summaries = []
    for country_name, cdf in out_df.groupby("country"):
        country_summaries.append(summarize_scope(cdf, country_name))

    country_summary_df = pd.DataFrame(country_summaries).sort_values(by="scope").reset_index(drop=True)
    save_csv(country_summary_df, country_summary_output_path)

    print(f"\nSaved Chroma evaluation file to: {output_path}")
    print(f"Rows: {len(out_df)}")
    print(f"Saved Chroma overall summary to: {summary_output_path}")
    print(f"Saved Chroma country summary to: {country_summary_output_path}")

    print_summary_block(overall_summary, "Chroma Average Metrics Summary:")

    print("\nChroma Average Metrics By Country:")
    if len(country_summary_df) > 0:
        print(
            country_summary_df[[
                "scope",
                "num_questions",
                "avg_response_match",
                "avg_response_match_normalized",
                "avg_score_abs_difference",
                "avg_cosine_similarity_sbert",
                "avg_rouge_l_f1",
                "avg_confidence",
            ]].to_string(index=False)
        )

    if len(out_df) > 0:
        print("\nPreview:")
        print(
            out_df[[
                "country", "question_id", "response_match",
                "response_match_normalized",
                "score_abs_difference", "cosine_similarity_minilm",
                "cosine_similarity_sbert", "tfidf_similarity",
                "jaccard_similarity", "bleu_score", "rouge_l_f1"
            ]].head(20).to_string(index=False)
        )


if __name__ == "__main__":
    main()