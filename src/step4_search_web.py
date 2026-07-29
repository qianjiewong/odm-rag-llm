import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from config import INTERIM_DIR
from utils import print_header, save_csv, ensure_dir

load_dotenv()

SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "exa").strip().lower()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
EXA_API_KEY = os.getenv("EXA_API_KEY", "").strip()

TAVILY_URL = "https://api.tavily.com/search"
EXA_URL = "https://api.exa.ai/search"


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def safe_console_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("cp1252", errors="replace").decode("cp1252"))


def normalize_country_domains(country: str) -> list[str]:
    """
    Optional domain hints for Exa.
    These help bias runtime search toward more official national sources.
    """
    country = safe_text(country).lower()

    domain_map = {
        "albania": ["gov.al"],
        "austria": ["gv.at", "data.gv.at"],
        "belgium": ["belgium.be", "data.gov.be"],
        "bosnia and herzegovina": ["gov.ba"],
        "bulgaria": ["government.bg", "egov.bg"],
        "croatia": ["gov.hr", "data.gov.hr"],
        "cyprus": ["gov.cy", "data.gov.cy"],
        "czechia": ["gov.cz", "data.gov.cz"],
        "denmark": ["gov.dk"],
        "estonia": ["gov.ee", "riik.ee", "avaandmed.eesti.ee"],
        "finland": ["gov.fi", "avoindata.fi"],
        "france": ["gouv.fr", "data.gouv.fr"],
        "germany": ["bund.de", "govdata.de"],
        "greece": ["gov.gr", "data.gov.gr"],
        "hungary": ["gov.hu"],
        "iceland": ["gov.is"],
        "ireland": ["gov.ie", "data.gov.ie"],
        "italy": ["gov.it", "dati.gov.it"],
        "latvia": ["gov.lv", "data.gov.lv"],
        "lithuania": ["gov.lt", "data.gov.lt"],
        "luxembourg": ["gouvernement.lu", "data.public.lu"],
        "malta": ["gov.mt", "data.gov.mt"],
        "montenegro": ["gov.me"],
        "netherlands": ["overheid.nl", "data.overheid.nl"],
        "north macedonia": ["gov.mk", "data.gov.mk"],
        "norway": ["regjeringen.no", "data.norge.no"],
        "poland": ["gov.pl", "dane.gov.pl"],
        "portugal": ["gov.pt", "dados.gov.pt"],
        "romania": ["gov.ro", "data.gov.ro"],
        "serbia": ["gov.rs", "data.gov.rs"],
        "slovakia": ["gov.sk", "data.gov.sk"],
        "slovenia": ["gov.si", "data.gov.si"],
        "spain": ["gob.es", "datos.gob.es"],
        "sweden": ["gov.se"],
        "switzerland": ["admin.ch", "opendata.swiss"],
        "ukraine": ["gov.ua", "data.gov.ua"],
    }

    return domain_map.get(country, [])


def shorten_query(text: str, max_len: int = 300) -> str:
    text = safe_text(text)
    if len(text) <= max_len:
        return text

    cut = text[:max_len]
    last_space = cut.rfind(" ")
    if last_space > 80:
        cut = cut[:last_space]

    return cut.strip()


def search_tavily(query: str, max_results: int = 8) -> list[dict]:
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY is missing in .env")

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
        "include_answer": False,
        "include_raw_content": False,
    }

    response = requests.post(TAVILY_URL, json=payload, timeout=60)
    response.raise_for_status()

    data = response.json()
    results = data.get("results", [])

    cleaned = []
    for rank, item in enumerate(results, start=1):
        cleaned.append({
            "rank": rank,
            "title": safe_text(item.get("title", "")),
            "url": safe_text(item.get("url", "")),
            "snippet": safe_text(item.get("content", "")),
            "published_date": safe_text(item.get("published_date", "")),
            "provider": "tavily",
        })
    return cleaned


def search_exa(query: str, max_results: int = 8, country: str = "") -> list[dict]:
    """
    Exa search using direct HTTP requests.

    Docs:
    - POST https://api.exa.ai/search
    - x-api-key header
    - JSON body supports query, numResults, includeDomains, contents, etc.
    """
    if not EXA_API_KEY:
        raise ValueError("EXA_API_KEY is missing in .env")

    query = shorten_query(query, max_len=300)

    headers = {
        "x-api-key": EXA_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "query": query,
        "numResults": max_results,
        "contents": {
            "highlights": True,
            "text": False,
        },
    }

    include_domains = normalize_country_domains(country)
    if include_domains:
        payload["includeDomains"] = include_domains

    response = requests.post(EXA_URL, json=payload, headers=headers, timeout=60)

    if not response.ok:
        raise RuntimeError(
            f"Exa search failed for query='{query}' "
            f"with status {response.status_code}: {response.text}"
        )

    data = response.json()
    results = data.get("results", [])

    cleaned = []

    for rank, item in enumerate(results, start=1):
        title = safe_text(item.get("title", ""))
        url = safe_text(item.get("url", ""))

        highlights = item.get("highlights", [])
        if isinstance(highlights, list) and highlights:
            snippet = " ".join([safe_text(x) for x in highlights[:3] if safe_text(x)])
        else:
            snippet = safe_text(item.get("text", ""))[:1200]

        published_date = safe_text(item.get("publishedDate", ""))

        cleaned.append({
            "rank": rank,
            "title": title,
            "url": url,
            "snippet": snippet,
            "published_date": published_date,
            "provider": "exa",
        })

    return cleaned


def search_web(query: str, max_results: int = 8, country: str = "") -> list[dict]:
    if SEARCH_PROVIDER == "exa":
        return search_exa(query=query, max_results=max_results, country=country)
    return search_tavily(query=query, max_results=max_results)


def search_web_for_question(
    country: str,
    question_id: str,
    queries_df: pd.DataFrame,
    output_dir: Path = None,
) -> pd.DataFrame:
    """
    Reusable single-question search helper for the UI/runtime pipeline.
    This now uses the same provider selection logic as the batch search path.
    """
    rows = []
    seen_urls = set()

    for _, row in queries_df.iterrows():
        query = safe_text(row.get("query", ""))
        query = shorten_query(query, max_len=300)

        try:
            results = search_web(
                query=query,
                max_results=6,
                country=country,
            )
        except Exception as e:
            raise RuntimeError(
                f"{SEARCH_PROVIDER.capitalize()} search failed for query='{query}' "
                f"with error: {e}"
            )

        for result in results:
            url = safe_text(result.get("url", ""))
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            rows.append({
                "country": country,
                "question_id": question_id,
                "query": query,
                "title": safe_text(result.get("title", "")),
                "url": url,
                "snippet": safe_text(result.get("snippet", "")),
                "published_date": safe_text(result.get("published_date", "")),
                "provider": safe_text(result.get("provider", SEARCH_PROVIDER)),
                "rank": result.get("rank", None),
            })

    out_df = pd.DataFrame(rows)

    if output_dir is not None:
        output_path = output_dir / "search_results.csv"
        out_df.to_csv(output_path, index=False)

    return out_df


def main():
    print_header("STEP 4: SEARCH WEB")

    input_path = INTERIM_DIR / "queries" / "generated_queries_odm_selected.csv"
    output_dir = INTERIM_DIR / "search_results"
    output_path = output_dir / "search_results_odm_selected.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Generated queries file not found: {input_path}\n"
            f"Please run step3_generate_queries.py first."
        )

    ensure_dir(output_dir)

    queries_df = pd.read_csv(input_path)
    rows = []

    for _, row in queries_df.iterrows():
        country = safe_text(row.get("country", ""))
        question_id = safe_text(row.get("question_id", ""))
        dimension = safe_text(row.get("dimension", ""))
        question = safe_text(row.get("question", ""))
        query_no = row.get("query_no", "")
        query = safe_text(row.get("query", ""))
        search_language = safe_text(row.get("search_language", ""))

        if not query:
            continue

        query = shorten_query(query, max_len=300)

        try:
            results = search_web(query=query, max_results=8, country=country)
        except Exception as e:
            safe_console_print(f"Search failed for {country} | {question_id} | query={query}")
            safe_console_print(str(e))
            results = []

        for result in results:
            rows.append({
                "country": country,
                "question_id": question_id,
                "dimension": dimension,
                "question": question,
                "query_no": query_no,
                "query": query,
                "search_language": search_language,
                "rank": result["rank"],
                "title": result["title"],
                "url": result["url"],
                "snippet": result["snippet"],
                "published_date": result["published_date"],
                "provider": result["provider"],
            })

        safe_console_print(
            f"Searched {SEARCH_PROVIDER}: {country} | {question_id} | query_no={query_no} | results={len(results)}"
        )
        time.sleep(0.2)

    out_df = pd.DataFrame(rows).drop_duplicates(subset=["country", "question_id", "query", "url"])

    save_csv(out_df, output_path)

    safe_console_print(f"\nSaved search results to: {output_path}")
    safe_console_print(f"Rows: {len(out_df)}")

    if len(out_df) > 0:
        preview = out_df.head(20).to_string(index=False)
        safe_console_print("\nPreview:")
        safe_console_print(preview)


if __name__ == "__main__":
    main()