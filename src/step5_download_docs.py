import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from config import INTERIM_DIR
from utils import print_header, save_csv, ensure_dir

USER_AGENT = "Mozilla/5.0 (compatible; ODM-RAG/1.0)"
REQUEST_TIMEOUT = 25
SLEEP_SECONDS = 0.15

MAX_URLS_PER_QUESTION = int(os.getenv("MAX_URLS_PER_QUESTION", "20"))
RUNTIME_MAX_URLS_PER_QUESTION = int(os.getenv("RUNTIME_MAX_URLS_PER_QUESTION", "8"))

SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".mp4", ".mp3", ".zip", ".rar", ".7z", ".ppt", ".pptx",
    ".doc", ".docx", ".xls", ".xlsx"
}

BLOCKED_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "twitter.com",
    "x.com",
    "youtube.com",
}

# Benchmark / answer-leakage blocking
BLOCKED_URL_PATTERNS = [
    "open-data-maturity",
    "country-factsheet",
    "factsheet",
    "self-assessment",
    "questionnaire",
    "2025_odm_questionnaire",
]

BLOCKED_TITLE_PATTERNS = [
    "open data maturity",
    "factsheet",
    "questionnaire",
    "self-assessment",
]

# New procurement / tender blocking
PROCUREMENT_BLOCKED_DOMAINS = {
    "riigihanked.riik.ee",
    "www.riigihanked.riik.ee",
    "ted.europa.eu",
    "simap.ted.europa.eu",
}

PROCUREMENT_URL_PATTERNS = [
    "riigihanked",
    "/rhr/",
    "procurement",
    "tender",
    "contract-award",
    "award-notice",
    "prior-information-notice",
    "call-for-tenders",
    "hanketeade",
    "riigihange",
]

PROCUREMENT_TITLE_PATTERNS = [
    "procurement notice",
    "contract award",
    "prior information notice",
    "call for tenders",
    "software development services",
    "hanketeade",
    "riigihange",
]


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", safe_text(text).lower()).strip("_")


def get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def is_blocked_answer_source(url: str, title: str) -> bool:
    url_l = safe_text(url).lower()
    title_l = safe_text(title).lower()

    if any(pattern in url_l for pattern in BLOCKED_URL_PATTERNS):
        return True

    if any(pattern in title_l for pattern in BLOCKED_TITLE_PATTERNS):
        return True

    return False


def is_procurement_like_source(url: str, title: str) -> bool:
    url_l = safe_text(url).lower()
    title_l = safe_text(title).lower()
    domain = get_domain(url_l)

    if domain in PROCUREMENT_BLOCKED_DOMAINS:
        return True

    if any(pattern in url_l for pattern in PROCUREMENT_URL_PATTERNS):
        return True

    if any(pattern in title_l for pattern in PROCUREMENT_TITLE_PATTERNS):
        return True

    return False


def should_skip_url(url: str) -> bool:
    url_l = safe_text(url).lower()
    domain = get_domain(url_l)

    if any(blocked in domain for blocked in BLOCKED_DOMAINS):
        return True

    for ext in SKIP_EXTENSIONS:
        if url_l.endswith(ext):
            return True

    return False


def detect_file_type(url: str, content_type: str) -> str:
    url_l = safe_text(url).lower()
    ct_l = safe_text(content_type).lower()

    if "pdf" in ct_l or url_l.endswith(".pdf"):
        return "pdf"
    return "html"


def download_file(url: str, out_path_without_suffix: Path) -> tuple[str, str]:
    headers = {"User-Agent": USER_AGENT}

    response = requests.get(
        url,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
        allow_redirects=True,
        stream=True,
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    file_type = detect_file_type(url, content_type)

    suffix = ".pdf" if file_type == "pdf" else ".html"
    final_path = out_path_without_suffix.with_suffix(suffix)

    with open(final_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return str(final_path), file_type


def build_download_rows_for_question(
    qdf: pd.DataFrame,
    country: str,
    question_id: str,
    base_out_dir: Path,
    max_urls_per_question: int,
    use_slug_filenames: bool = True,
    sleep_seconds: float = SLEEP_SECONDS,
    verbose: bool = False,
) -> tuple[list[dict], int, int]:
    """
    Shared download logic for BOTH:
    - batch evaluation path
    - UI runtime path

    Returns:
    - rows: downloaded document metadata rows
    - blocked_answer_sources_count
    - blocked_procurement_sources_count
    """
    rows = []
    seen_urls = set()
    blocked_answer_sources = 0
    blocked_procurement_sources = 0
    downloaded_for_question = 0

    qdf = qdf.copy()

    sort_cols = [c for c in ["query_no", "rank"] if c in qdf.columns]
    if sort_cols:
        qdf = qdf.sort_values(by=sort_cols)

    for _, row in qdf.iterrows():
        if downloaded_for_question >= max_urls_per_question:
            break

        url = safe_text(row.get("url", ""))
        title = safe_text(row.get("title", ""))
        dimension = safe_text(row.get("dimension", ""))
        question = safe_text(row.get("question", ""))
        query_no = row.get("query_no", row.get("query_rank", ""))
        query = safe_text(row.get("query", ""))

        if not url or url in seen_urls:
            continue

        if is_blocked_answer_source(url, title):
            blocked_answer_sources += 1
            if verbose:
                print(f"Blocked answer-like source: {country} | {question_id} | {title} | {url}")
            continue

        if is_procurement_like_source(url, title):
            blocked_procurement_sources += 1
            if verbose:
                print(f"Blocked procurement-like source: {country} | {question_id} | {title} | {url}")
            continue

        if should_skip_url(url):
            if verbose:
                print(f"Skipped file/domain: {country} | {question_id} | {url}")
            continue

        seen_urls.add(url)

        if use_slug_filenames:
            file_stub = f"{slugify(country)}_{slugify(question_id)}_{downloaded_for_question + 1}"
        else:
            file_stub = f"doc_{downloaded_for_question + 1}"

        out_base = base_out_dir / file_stub

        try:
            saved_path, file_type = download_file(url, out_base)

            rows.append({
                "country": country,
                "question_id": question_id,
                "dimension": dimension,
                "question": question,
                "query_no": query_no,
                "query": query,
                "title": title,
                "url": url,
                "saved_path": saved_path,
                "file_type": file_type,
            })

            downloaded_for_question += 1

            if verbose:
                print(f"Downloaded: {country} | {question_id} | {file_type} | {url}")

        except Exception as e:
            if verbose:
                print(f"Error: {country} | {question_id} | {url}")
                print(f"Error detail: {e}")

        time.sleep(sleep_seconds)

    return rows, blocked_answer_sources, blocked_procurement_sources


def download_docs_for_question(
    country: str,
    question_id: str,
    search_df: pd.DataFrame,
    output_dir: Path = None,
) -> pd.DataFrame:
    """
    Reusable single-question downloader for UI runtime path.
    Uses the SAME filtering + download logic as the batch evaluation path.
    """
    if output_dir is None:
        raise ValueError("output_dir is required for runtime document download")

    downloads_dir = output_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    rows, _, _ = build_download_rows_for_question(
        qdf=search_df,
        country=country,
        question_id=question_id,
        base_out_dir=downloads_dir,
        max_urls_per_question=RUNTIME_MAX_URLS_PER_QUESTION,
        use_slug_filenames=False,
        sleep_seconds=0.0,
        verbose=False,
    )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_dir / "downloaded_docs.csv", index=False)
    return out_df


def main():
    print_header("STEP 5: DOWNLOAD DOCUMENTS")

    input_path = INTERIM_DIR / "search_results" / "search_results_odm_selected.csv"
    output_dir = INTERIM_DIR / "downloaded_docs"
    output_path = output_dir / "download_index_odm_selected.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Search results file not found: {input_path}\n"
            f"Please run step4_search_web.py first."
        )

    ensure_dir(output_dir)

    pdf_dir = output_dir / "pdf"
    ensure_dir(pdf_dir)

    df = pd.read_csv(input_path)

    rows = []
    blocked_answer_sources_total = 0
    blocked_procurement_sources_total = 0

    for (country, question_id), qdf in df.groupby(["country", "question_id"]):
        question_rows, blocked_count, procurement_blocked_count = build_download_rows_for_question(
            qdf=qdf,
            country=country,
            question_id=question_id,
            base_out_dir=pdf_dir,
            max_urls_per_question=MAX_URLS_PER_QUESTION,
            use_slug_filenames=True,
            sleep_seconds=SLEEP_SECONDS,
            verbose=True,
        )

        rows.extend(question_rows)
        blocked_answer_sources_total += blocked_count
        blocked_procurement_sources_total += procurement_blocked_count

    out_df = pd.DataFrame(rows)
    save_csv(out_df, output_path)

    print(f"\nSaved download index to: {output_path}")
    print(f"Rows: {len(out_df)}")
    print(f"Blocked answer-like sources: {blocked_answer_sources_total}")
    print(f"Blocked procurement-like sources: {blocked_procurement_sources_total}")

    if len(out_df) > 0:
        print("\nPreview:")
        print(out_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()