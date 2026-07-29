from pathlib import Path
import pandas as pd

from config import INTERIM_DIR, GENERATION_DIR, PROCESSED_DIR


def safe_text(value):
    return str(value or "").strip()


def filter_df(df, country=None, question_id=None):
    out = df.copy()
    if country is not None and "country" in out.columns:
        out = out[out["country"].astype(str).str.strip() == country]
    if question_id is not None and "question_id" in out.columns:
        out = out[out["question_id"].astype(str).str.strip() == question_id]
    return out


def main():
    country = "Estonia"
    question_id = None  # set like "P1" or "I2" if you want one specific question

    files = {
        "ground_truth": PROCESSED_DIR / "ground_truth_odm_selected.csv",
        "queries": INTERIM_DIR / "queries" / "generated_queries_odm_selected.csv",
        "search_results": INTERIM_DIR / "search_results" / "search_results_odm_selected.csv",
        "download_index": INTERIM_DIR / "downloaded_docs" / "download_index_odm_selected.csv",
        "extracted_text": INTERIM_DIR / "extracted_text" / "extracted_text_odm_selected.csv",
        "chunks": INTERIM_DIR / "chunks" / "chunks_odm_selected.csv",
        "retrieval": INTERIM_DIR / "retrieval" / "retrieved_topk_odm_selected.csv",
        "generated_answers": GENERATION_DIR / "generated_answers_odm_selected.csv",
    }

    out_dir = Path("debug_country_inspection")
    out_dir.mkdir(exist_ok=True)

    for name, path in files.items():
        if not path.exists():
            print(f"Missing file: {path}")
            continue

        df = pd.read_csv(path)
        filtered = filter_df(df, country=country, question_id=question_id)

        out_path = out_dir / f"{name}_{country.lower()}.csv"
        filtered.to_csv(out_path, index=False)
        print(f"Saved {name}: {len(filtered)} rows -> {out_path}")


if __name__ == "__main__":
    main()