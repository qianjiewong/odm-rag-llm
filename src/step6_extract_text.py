from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from config import INTERIM_DIR
from utils import print_header, save_csv, ensure_dir


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_text(text: str) -> str:
    text = str(text or "")
    text = " ".join(text.split())
    return text.strip()


def extract_text_from_html(path: str) -> str:
    """
    Shared HTML extraction logic for BOTH:
    - evaluation batch path
    - UI runtime path
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return clean_text(text)
    except Exception:
        return ""


def extract_text_from_pdf(path: str) -> str:
    """
    Shared PDF extraction logic for BOTH:
    - evaluation batch path
    - UI runtime path

    Uses pypdf first, then falls back to PyPDF2.
    """
    text = ""

    try:
        from pypdf import PdfReader

        reader = PdfReader(path)
        for page in reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text += " " + safe_text(page_text)
            except Exception:
                continue

        cleaned = clean_text(text)
        if cleaned:
            return cleaned
    except Exception:
        pass

    try:
        import PyPDF2

        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += " " + safe_text(page_text)
                except Exception:
                    continue

        return clean_text(text)
    except Exception:
        return ""


def extract_text_by_file_type(saved_path: str, file_type: str) -> str:
    file_type = safe_text(file_type).lower()

    if file_type == "html":
        return extract_text_from_html(saved_path)
    if file_type == "pdf":
        return extract_text_from_pdf(saved_path)

    return ""


def build_extracted_rows(
    df: pd.DataFrame,
    include_query_metadata: bool = True,
    verbose: bool = False,
) -> list[dict]:
    """
    Shared extraction core for BOTH:
    - evaluation batch path
    - UI runtime path

    Parameters
    ----------
    df : pd.DataFrame
        Input download index rows.
    include_query_metadata : bool
        If True, includes dimension/question/query/query_no in output when present.
    verbose : bool
        If True, prints extraction progress messages.

    Returns
    -------
    list[dict]
        Extracted text rows.
    """
    rows = []

    for _, row in df.iterrows():
        country = safe_text(row.get("country", ""))
        question_id = safe_text(row.get("question_id", ""))
        dimension = safe_text(row.get("dimension", ""))
        question = safe_text(row.get("question", ""))
        query_no = row.get("query_no", row.get("query_rank", ""))
        query = safe_text(row.get("query", ""))
        title = safe_text(row.get("title", ""))
        url = safe_text(row.get("url", ""))
        saved_path = safe_text(row.get("saved_path", ""))
        file_type = safe_text(row.get("file_type", ""))

        if not saved_path or not Path(saved_path).exists():
            if verbose:
                print(f"Missing file: {saved_path}")
            continue

        extracted_text = extract_text_by_file_type(saved_path, file_type)

        if not extracted_text:
            if verbose:
                print(f"Empty extracted text: {country} | {question_id} | {saved_path}")
            continue

        out_row = {
            "country": country,
            "question_id": question_id,
            "title": title,
            "url": url,
            "saved_path": saved_path,
            "file_type": file_type,
            "extracted_text": extracted_text,
            "char_count": len(extracted_text),
        }

        if include_query_metadata:
            out_row.update({
                "dimension": dimension,
                "question": question,
                "query_no": query_no,
                "query": query,
            })

        rows.append(out_row)

        if verbose:
            print(f"Extracted: {country} | {question_id} | {file_type} | chars={len(extracted_text)}")

    return rows


def extract_text_for_question(
    country: str,
    question_id: str,
    downloaded_df: pd.DataFrame,
    output_dir: Path = None,
) -> pd.DataFrame:
    """
    Reusable single-question extractor for UI runtime path.
    Now uses the SAME extraction logic as the batch evaluation path.
    """
    rows = build_extracted_rows(
        df=downloaded_df,
        include_query_metadata=True,
        verbose=False,
    )

    out_df = pd.DataFrame(rows)

    if output_dir is not None:
        out_df.to_csv(output_dir / "extracted_text.csv", index=False)

    return out_df


def main():
    print_header("STEP 6: EXTRACT TEXT")

    input_path = INTERIM_DIR / "downloaded_docs" / "download_index_odm_selected.csv"
    output_dir = INTERIM_DIR / "extracted_text"
    output_path = output_dir / "extracted_text_odm_selected.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Download index file not found: {input_path}\n"
            f"Please run step5_download_docs.py first."
        )

    ensure_dir(output_dir)

    df = pd.read_csv(input_path)

    rows = build_extracted_rows(
        df=df,
        include_query_metadata=True,
        verbose=True,
    )

    out_df = pd.DataFrame(rows)
    save_csv(out_df, output_path)

    print(f"\nSaved extracted text to: {output_path}")
    print(f"Rows: {len(out_df)}")

    if len(out_df) > 0:
        print("\nPreview:")
        print(
            out_df[[
                "country", "question_id", "file_type", "char_count", "title"
            ]].head(20).to_string(index=False)
        )


if __name__ == "__main__":
    main()