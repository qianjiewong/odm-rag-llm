import os
import json
import ast
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
TABLE_NAME = "search_history"


def safe_text(value) -> str:
    return str(value or "").strip()


def parse_response_scoring(raw: str) -> dict:
    raw = safe_text(raw)
    if not raw:
        return {}

    # First try JSON
    try:
        parsed = json.loads(raw)
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

    # Then try Python dict string style
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


def load_registry_response_scoring_map() -> dict:
    """
    Build a lookup:
        question_id -> parsed response_scoring dict
    """
    base_dir = Path(__file__).resolve().parent.parent
    processed_dir = base_dir / "data" / "processed"

    registry_path = processed_dir / "registry_odm_demo_policy_impact.csv"
    fallback_path = processed_dir / "registry_odm_selected_questions.csv"

    if registry_path.exists():
        df = pd.read_csv(registry_path)
    elif fallback_path.exists():
        df = pd.read_csv(fallback_path)
    else:
        return {}

    if "question_id" not in df.columns:
        return {}

    lookup = {}

    for _, row in df.iterrows():
        qid = safe_text(row.get("question_id", ""))
        scoring_raw = safe_text(row.get("response_scoring", ""))
        lookup[qid] = parse_response_scoring(scoring_raw)

    return lookup


RESPONSE_SCORING_LOOKUP = load_registry_response_scoring_map()


def get_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL or SUPABASE_KEY is missing in .env")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def init_db():
    client = get_client()
    client.table(TABLE_NAME).select("id").limit(1).execute()


def normalize_dimension_label(value: str) -> str:
    raw = str(value or "").strip().lower()

    if raw == "policy_dimension":
        return "Policy"
    if raw == "impact_dimension":
        return "Impact"
    if raw == "policy":
        return "Policy"
    if raw == "impact":
        return "Impact"

    return str(value or "").strip()


def save_search_result(result: dict) -> int:
    client = get_client()

    payload = {
        "mode": result.get("mode", ""),
        "country": result.get("country", ""),
        "question_id": result.get("question_id", ""),
        "dimension": normalize_dimension_label(result.get("dimension", "")),
        "question": result.get("question", ""),
        "response_scoring": json.dumps(result.get("response_scoring", {}) or {}),
        "final_answer_selected": result.get("final_answer_selected", ""),
        "score_suggestion": result.get("score_suggestion", None),
        "confidence": result.get("confidence", None),
        "evidence_sufficiency": result.get("evidence_sufficiency", ""),
        "justification_generated": result.get("justification_generated", ""),
        "retrieved_sources_json": result.get("retrieved_sources", []),
    }

    response = client.table(TABLE_NAME).insert(payload).execute()

    if not response.data:
        raise ValueError("Supabase insert failed: no data returned")

    return int(response.data[0]["id"])


def _normalize_saved_row(row: dict) -> dict:
    response_scoring_raw = row.get("response_scoring", "{}")
    response_scoring = parse_response_scoring(response_scoring_raw)

    # Fallback for older rows that do not store response_scoring yet
    if not response_scoring:
        qid = safe_text(row.get("question_id", ""))
        response_scoring = RESPONSE_SCORING_LOOKUP.get(qid, {})

    return {
        "id": row.get("id"),
        "created_at": row.get("created_at"),
        "mode": row.get("mode", ""),
        "country": row.get("country", ""),
        "question_id": row.get("question_id", ""),
        "dimension": normalize_dimension_label(row.get("dimension", "")),
        "question": row.get("question", ""),
        "response_scoring": response_scoring,
        "final_answer_selected": row.get("final_answer_selected", ""),
        "score_suggestion": row.get("score_suggestion", None),
        "confidence": row.get("confidence", None),
        "evidence_sufficiency": row.get("evidence_sufficiency", ""),
        "justification_generated": row.get("justification_generated", ""),
        "retrieved_sources": row.get("retrieved_sources_json", []) or [],
    }


def list_saved_results_filtered(
    country: Optional[str] = None,
    question_id: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    client = get_client()

    query = (
        client.table(TABLE_NAME)
        .select("id, created_at, country, question_id, dimension, question")
        .order("created_at", desc=False)
        .limit(limit)
    )

    if country:
        query = query.eq("country", country)

    if question_id:
        query = query.eq("question_id", question_id)

    response = query.execute()
    return response.data or []


def get_saved_result(record_id: int) -> Optional[dict]:
    client = get_client()

    response = (
        client.table(TABLE_NAME)
        .select("*")
        .eq("id", record_id)
        .limit(1)
        .execute()
    )

    if not response.data:
        return None

    return _normalize_saved_row(response.data[0])


def find_latest_saved_result(country: str, question_id: str) -> Optional[dict]:
    client = get_client()

    response = (
        client.table(TABLE_NAME)
        .select("*")
        .eq("country", country)
        .eq("question_id", question_id)
        .order("id", desc=True)
        .limit(1)
        .execute()
    )

    if not response.data:
        return None

    return _normalize_saved_row(response.data[0])


def find_latest_saved_custom_result(country: str, question_text: str) -> Optional[dict]:
    client = get_client()

    response = (
        client.table(TABLE_NAME)
        .select("*")
        .eq("country", country)
        .eq("question_id", "CUSTOM")
        .eq("question", question_text)
        .order("id", desc=True)
        .limit(1)
        .execute()
    )

    if not response.data:
        return None

    return _normalize_saved_row(response.data[0])


def delete_saved_result(record_id: int) -> None:
    client = get_client()
    client.table(TABLE_NAME).delete().eq("id", record_id).execute()