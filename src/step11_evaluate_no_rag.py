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
    print_header("STEP 11 (NO-RAG): EVALUATE ODM OUTPUTS")

    ground_truth_path = PROCESSED_DIR / "ground_truth_odm_selected.csv"
    generated_path = GENERATION_DIR / "generated_answers_odm_selected_no_rag.csv"

    output_dir = EVALUATION_DIR
    output_path = output_dir / "evaluation_odm_selected_no_rag.csv"
    summary_output_path = output_dir / "evaluation_summary_odm_selected_no_rag.csv"
    country_summary_output_path = output_dir / "evaluation_summary_by_country_odm_selected_no_rag.csv"

    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {ground_truth_path}\n"
            f"Please run step2_build_ground_truth.py first."
        )

    if not generated_path.exists():
        raise FileNotFoundError(
            f"Generated no-RAG answers file not found: {generated_path}\n"
            f"Please run step10_generate_answers_no_rag.py first."
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

        ground_truth_score = safe_float(row.get("awarded_score", None))
        generated_score = safe_float(row.get("score_suggestion", None))

        confidence = safe_float(row.get("confidence", None))
        evidence_sufficiency = safe_text(row.get("evidence_sufficiency", ""))

        response_match = int(ground_truth_answer == generated_answer)
        response_match_normalized = int(
            normalize_text(ground_truth_answer) == normalize_text(generated_answer)
        )

        if ground_truth_score is not None and generated_score is not None:
            score_difference = generated_score - ground_truth_score
            score_abs_difference = abs(score_difference)
        else:
            score_difference = None
            score_abs_difference = None

        if ground_truth_explanation and generated_justification:
            minilm_emb = minilm_model.encode(
                [ground_truth_explanation, generated_justification],
                convert_to_numpy=True,
                normalize_embeddings=False
            )
            cosine_similarity_minilm = cosine_from_vectors(minilm_emb[0], minilm_emb[1])

            sbert_emb = sbert_model.encode(
                [ground_truth_explanation, generated_justification],
                convert_to_numpy=True,
                normalize_embeddings=False
            )
            cosine_similarity_sbert = cosine_from_vectors(sbert_emb[0], sbert_emb[1])

            tfidf_similarity = tfidf_cosine_similarity(
                ground_truth_explanation, generated_justification
            )
            jaccard = jaccard_similarity(
                ground_truth_explanation, generated_justification
            )
            bleu = bleu_score(
                ground_truth_explanation, generated_justification
            )
            rouge_l = rouge_l_f1(
                ground_truth_explanation, generated_justification
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
            "response_match": response_match,
            "response_match_normalized": response_match_normalized,
            "ground_truth_score": ground_truth_score,
            "generated_score": generated_score,
            "score_difference": score_difference,
            "score_abs_difference": score_abs_difference,
            "ground_truth_explanation": ground_truth_explanation,
            "generated_justification": generated_justification,
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
            f"Evaluated no-RAG {country} | {question_id} | "
            f"match={response_match_normalized} | "
            f"abs_score_diff={score_abs_difference}"
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

    print(f"\nSaved no-RAG evaluation file to: {output_path}")
    print(f"Rows: {len(out_df)}")
    print(f"Saved overall summary to: {summary_output_path}")
    print(f"Saved country summary to: {country_summary_output_path}")

    print_summary_block(overall_summary, "No-RAG Average Metrics Summary:")

    print("\nNo-RAG Average Metrics By Country:")
    if len(country_summary_df) > 0:
        print(
            country_summary_df[[
                "scope",
                "num_questions",
                "avg_response_match_normalized",
                "avg_score_abs_difference",
                "avg_cosine_similarity_sbert",
                "avg_rouge_l_f1",
                "avg_confidence",
            ]].to_string(index=False)
        )


if __name__ == "__main__":
    main()