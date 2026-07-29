import ast
import pandas as pd

from config import PROCESSED_DIR, COUNTRY_FILE_MAP, SELECTED_QUESTION_IDS
from utils import print_header, save_csv, ensure_dir


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_response_scoring(raw: str) -> dict:
    raw = safe_text(raw)
    if not raw:
        return {}

    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            cleaned = {}
            for k, v in parsed.items():
                key = safe_text(k)
                try:
                    cleaned[key] = float(v)
                except Exception:
                    cleaned[key] = v
            return cleaned
    except Exception:
        pass

    return {}


def infer_awarded_score(final_answer: str, response_scoring: str):
    final_answer = safe_text(final_answer)
    score_map = parse_response_scoring(response_scoring)

    if not final_answer or not score_map:
        return None

    for option, score in score_map.items():
        if final_answer.lower() == safe_text(option).lower():
            return score

    return None


def load_selected_questions_from_sheet(country: str, file_path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    df = df.iloc[:, :7].copy()
    df.columns = [
        "question_id",
        "question",
        "answer_2024",
        "explanation_2024",
        "review_note",
        "updated_answer",
        "updated_explanation",
    ]

    rows = []

    for _, row in df.iterrows():
        qid = safe_text(row["question_id"])

        if qid not in SELECTED_QUESTION_IDS:
            continue

        final_answer = safe_text(row["updated_answer"]) or safe_text(row["answer_2024"])
        final_explanation = safe_text(row["updated_explanation"]) or safe_text(row["explanation_2024"])

        rows.append({
            "country": country,
            "question_id": qid,
            "question": safe_text(row["question"]),
            "answer_2024": safe_text(row["answer_2024"]),
            "explanation_2024": safe_text(row["explanation_2024"]),
            "review_note": safe_text(row["review_note"]),
            "updated_answer": safe_text(row["updated_answer"]),
            "updated_explanation": safe_text(row["updated_explanation"]),
            "final_answer": final_answer,
            "final_explanation": final_explanation,
        })

    return pd.DataFrame(rows)


def main():
    print_header("STEP 2: BUILD ODM SELECTED GROUND TRUTH")

    ensure_dir(PROCESSED_DIR)

    registry_path = PROCESSED_DIR / "registry_odm_selected_questions.csv"
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Registry file not found: {registry_path}\n"
            f"Please run step1_build_odm_policy_registry.py first."
        )

    registry_df = pd.read_csv(registry_path)
    registry_df = registry_df.drop_duplicates(subset=["question_id"]).copy()

    all_rows = []

    for country, file_path in COUNTRY_FILE_MAP.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing questionnaire for {country}: {file_path}")

        policy_df = load_selected_questions_from_sheet(country, file_path, "Policy")
        impact_df = load_selected_questions_from_sheet(country, file_path, "Impact")

        country_df = pd.concat([policy_df, impact_df], ignore_index=True)

        missing = sorted(set(SELECTED_QUESTION_IDS) - set(country_df["question_id"].tolist()))
        if missing:
            print(f"Warning: {country} missing selected questions: {missing}")

        all_rows.append(country_df)

    out_df = pd.concat(all_rows, ignore_index=True)

    out_df["dimension"] = out_df["question_id"].apply(
        lambda x: "Policy" if str(x).startswith("P") else "Impact"
    )

    out_df = out_df.merge(
        registry_df[["question_id", "response_scoring"]],
        on="question_id",
        how="left"
    )

    out_df["awarded_score"] = out_df.apply(
        lambda r: infer_awarded_score(
            r.get("final_answer", ""),
            r.get("response_scoring", "")
        ),
        axis=1
    )

    out_df = out_df.sort_values(by=["country", "question_id"]).reset_index(drop=True)

    output_path = PROCESSED_DIR / "ground_truth_odm_selected.csv"
    save_csv(out_df, output_path)

    print(f"Saved ground truth to: {output_path}")
    print(f"Rows: {len(out_df)}")
    print("\nPreview:")
    print(
        out_df[[
            "country", "question_id", "final_answer",
            "response_scoring", "awarded_score"
        ]].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()