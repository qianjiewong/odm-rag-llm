import pandas as pd
from pathlib import Path

from config import INTERIM_DIR
from utils import print_header, save_csv, ensure_dir


CHUNK_SIZE = 180
CHUNK_OVERLAP = 40


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_text(text: str) -> str:
    text = str(text or "")
    text = " ".join(text.split())
    return text.strip()


def chunk_text_by_words(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Shared chunking logic for BOTH:
    - evaluation batch path
    - UI runtime path
    """
    text = clean_text(text)
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

        start = max(0, end - overlap)

    return chunks


def build_chunk_rows(
    df: pd.DataFrame,
    include_query_metadata: bool = True,
    verbose: bool = False,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Shared chunk-building core for BOTH:
    - evaluation batch path
    - UI runtime path

    Parameters
    ----------
    df : pd.DataFrame
        Extracted-text rows.
    include_query_metadata : bool
        Whether to preserve dimension/question/query/query_no in output.
    verbose : bool
        Whether to print progress lines.
    chunk_size : int
        Number of words per chunk.
    overlap : int
        Number of overlapping words between adjacent chunks.
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
        extracted_text = safe_text(row.get("extracted_text", ""))

        chunks = chunk_text_by_words(
            extracted_text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        if not chunks:
            if verbose:
                print(f"No chunks created: {country} | {question_id} | {title}")
            continue

        for i, chunk in enumerate(chunks, start=1):
            out_row = {
                "country": country,
                "question_id": question_id,
                "title": title,
                "url": url,
                "saved_path": saved_path,
                "file_type": file_type,
                "chunk_id": f"{country}_{question_id}_{i}",
                "chunk_index": i,
                "chunk_text": chunk,
                "chunk_length_words": len(chunk.split()),
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
            print(f"Chunked: {country} | {question_id} | {len(chunks)} chunks | {title}")

    return rows


def chunk_text_for_question(
    country: str,
    question_id: str,
    extracted_df: pd.DataFrame,
    output_dir: Path = None,
) -> pd.DataFrame:
    """
    Reusable single-question chunker for UI runtime path.
    Now uses the SAME chunking logic as the batch evaluation path.
    """
    rows = build_chunk_rows(
        df=extracted_df,
        include_query_metadata=True,
        verbose=False,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    out_df = pd.DataFrame(rows)

    if output_dir is not None:
        out_df.to_csv(output_dir / "chunks.csv", index=False)

    return out_df


def main():
    print_header("STEP 7: CHUNK EXTRACTED TEXT")

    input_path = INTERIM_DIR / "extracted_text" / "extracted_text_odm_selected.csv"
    output_dir = INTERIM_DIR / "chunks"
    output_path = output_dir / "chunks_odm_selected.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Extracted text file not found: {input_path}\n"
            f"Please run step6_extract_text.py first."
        )

    ensure_dir(output_dir)

    df = pd.read_csv(input_path)

    rows = build_chunk_rows(
        df=df,
        include_query_metadata=True,
        verbose=True,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    out_df = pd.DataFrame(rows)
    save_csv(out_df, output_path)

    print(f"\nSaved chunks to: {output_path}")
    print(f"Rows: {len(out_df)}")

    if len(out_df) > 0:
        print("\nPreview:")
        print(
            out_df[[
                "country", "question_id", "chunk_index", "chunk_length_words", "title"
            ]].head(20).to_string(index=False)
        )


if __name__ == "__main__":
    main()