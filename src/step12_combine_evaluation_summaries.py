from pathlib import Path
import pandas as pd

from config import EVALUATION_DIR
from utils import print_header, save_csv, ensure_dir


def main():
    print_header("STEP 12: COMBINE EVALUATION SUMMARIES")

    experiments_dir = EVALUATION_DIR / "experiments"
    output_path = EVALUATION_DIR / "combined_experiment_summaries.csv"

    if not experiments_dir.exists():
        raise FileNotFoundError(
            f"Experiments folder not found: {experiments_dir}\n"
            f"Please create it and place copied summary CSV files inside."
        )

    csv_files = sorted(experiments_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No experiment summary CSV files found in: {experiments_dir}"
        )

    rows = []

    for file_path in csv_files:
        df = pd.read_csv(file_path)

        if df.empty:
            print(f"Skipped empty file: {file_path.name}")
            continue

        row = df.iloc[0].to_dict()
        row["experiment_name"] = file_path.stem
        rows.append(row)

    out_df = pd.DataFrame(rows)

    if "experiment_name" in out_df.columns:
        cols = ["experiment_name"] + [c for c in out_df.columns if c != "experiment_name"]
        out_df = out_df[cols]

    save_csv(out_df, output_path)

    print(f"Saved combined summaries to: {output_path}")
    print(f"Rows: {len(out_df)}")

    if len(out_df) > 0:
        print("\nPreview:")
        print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()