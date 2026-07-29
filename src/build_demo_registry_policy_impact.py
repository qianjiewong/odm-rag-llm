from pathlib import Path
import pandas as pd

from config import RAW_DIR, PROCESSED_DIR


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_colname(name: str) -> str:
    return safe_text(name).lower().replace("\n", " ").replace("-", "_").replace(" ", "_")


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized_map = {normalize_colname(col): col for col in df.columns}
    for cand in candidates:
        cand_norm = normalize_colname(cand)
        if cand_norm in normalized_map:
            return normalized_map[cand_norm]
    return None


def main():
    input_path = RAW_DIR / "questionnaires" / "2025_odm_questionnaire_data.xlsx"
    output_path = PROCESSED_DIR / "registry_odm_demo_policy_impact.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Input questionnaire file not found: {input_path}")

    # Load first sheet
    df = pd.read_excel(input_path)

    # Try to find the key columns robustly
    question_id_col = find_column(df, [
        "question_id", "question id", "code", "question_code"
    ])
    dimension_col = find_column(df, [
        "dimension", "dimension_name"
    ])
    indicator_col = find_column(df, [
        "indicator", "indicator_name"
    ])
    question_col = find_column(df, [
        "question", "question_text", "text"
    ])
    response_scoring_col = find_column(df, [
        "response_scoring", "response scoring", "scoring", "score_map"
    ])

    if not question_id_col or not dimension_col or not question_col:
        raise ValueError(
            "Could not detect required columns. "
            f"Found columns: {list(df.columns)}"
        )

    out_df = pd.DataFrame({
        "question_id": df[question_id_col].map(safe_text),
        "dimension": df[dimension_col].map(safe_text),
        "indicator": df[indicator_col].map(safe_text) if indicator_col else "",
        "question": df[question_col].map(safe_text),
        "response_scoring": df[response_scoring_col].map(safe_text) if response_scoring_col else "",
    })

    # Keep only Policy + Impact
    out_df = out_df[
        out_df["dimension"].str.lower().isin(["policy", "impact", "policy_dimension", "impact_dimension"])
    ].copy()

    # Normalize dimension labels
    out_df["dimension"] = out_df["dimension"].replace({
        "policy_dimension": "Policy",
        "impact_dimension": "Impact",
        "policy": "Policy",
        "impact": "Impact",
    })

    # Remove blanks and duplicates
    out_df = out_df[
        (out_df["question_id"] != "") &
        (out_df["question"] != "")
    ].copy()

    out_df = out_df.drop_duplicates(subset=["question_id"]).sort_values(
        by=["dimension", "question_id"]
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"Saved demo registry to: {output_path}")
    print(f"Rows: {len(out_df)}")
    print("\nPreview:")
    print(out_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()