import pandas as pd

from config import (
    PROCESSED_DIR,
    SELECTED_QUESTION_IDS,
    COUNTRY_FILE_MAP,
    MASTER_QUESTIONNAIRE_FILE,
)
from utils import print_header, save_csv, ensure_dir


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def main():
    print_header("STEP 1: BUILD ODM SELECTED QUESTION REGISTRY")

    ensure_dir(PROCESSED_DIR)

    for country, path in COUNTRY_FILE_MAP.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing questionnaire for {country}: {path}")

    if not MASTER_QUESTIONNAIRE_FILE.exists():
        raise FileNotFoundError(
            f"Master questionnaire file not found: {MASTER_QUESTIONNAIRE_FILE}"
        )

    df = pd.read_excel(MASTER_QUESTIONNAIRE_FILE, sheet_name="questionnaire")

    required_cols = ["question_id", "dimension", "indicator", "question", "response_scoring"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in questionnaire sheet: {missing}")

    df = df[required_cols].copy()
    df["question_id"] = df["question_id"].astype(str).str.strip()

    registry_df = df[df["question_id"].isin(SELECTED_QUESTION_IDS)].copy()

    if registry_df.empty:
        raise ValueError("No selected questions were found in the master questionnaire.")

    missing_qids = sorted(set(SELECTED_QUESTION_IDS) - set(registry_df["question_id"].tolist()))
    if missing_qids:
        print(f"Warning: selected question IDs missing from questionnaire: {missing_qids}")

    registry_df["dimension"] = registry_df["dimension"].apply(safe_text)
    registry_df["indicator"] = registry_df["indicator"].apply(safe_text)
    registry_df["question"] = registry_df["question"].apply(safe_text)
    registry_df["response_scoring"] = registry_df["response_scoring"].apply(safe_text)

    registry_df = registry_df.sort_values(by="question_id").reset_index(drop=True)

    output_path = PROCESSED_DIR / "registry_odm_selected_questions.csv"
    save_csv(registry_df, output_path)

    print(f"Saved registry to: {output_path}")
    print(f"Rows: {len(registry_df)}")
    print("\nPreview:")
    print(
        registry_df[["question_id", "dimension", "indicator", "response_scoring"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()