import re
from pathlib import Path
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_text(value) -> str:
    if pd.isna(value) or value is None:
        return ""
    return str(value).strip()


def country_slug(country: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", country.lower()).strip("_")


def safe_console_print(text: str) -> None:
    text = str(text)
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("cp1252", errors="replace").decode("cp1252"))


def load_excel_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name)


def normalize_question_id(qid: str) -> str:
    return safe_text(qid)


def is_scored_policy_question(question_id: str) -> bool:
    """
    ODM methodology says policy has 28 scored questions.
    The workbook includes some helper/non-scored rows.
    We exclude:
    - P14 (descriptive governance type)
    - P26-a
    - P26-b
    """
    qid = normalize_question_id(question_id)

    excluded = {"P14", "P26-a", "P26-b"}
    return qid.startswith("P") and qid not in excluded