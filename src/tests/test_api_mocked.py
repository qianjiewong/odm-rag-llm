"""
Unit tests for the API-calling components of the pipeline, using mocking to
avoid real network calls to the OpenAI API. These tests verify the pipeline's
own logic (prompt construction, response parsing, fallback behaviour,
constrained response selection) without depending on live model output,
which is non-deterministic and would otherwise make these tests unreliable
and costly to run repeatedly.
"""

import sys
import os
import json
from unittest.mock import MagicMock

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from step3_generate_queries import generate_queries_core
from step10_generate_answers_chroma import generate_answer_for_question


def make_mock_openai_response(output_text: str):
    """Builds a mock object matching the shape of OpenAI's Responses API output."""
    mock_response = MagicMock()
    mock_response.output_text = output_text
    return mock_response


# ---------------------------------------------------------------------------
# Query generation (Section 5.2.1)
# ---------------------------------------------------------------------------

def test_generate_queries_core_returns_cleaned_llm_output():
    client = MagicMock()
    client.responses.create.return_value = make_mock_openai_response(
        json.dumps({"queries": [
            "France open data strategy",
            "France open data law",
            "France ministry digital strategy",
            "France national open data plan",
        ]})
    )

    queries = generate_queries_core(
        client=client, country="France", question_id="P2",
        question="Is there a national open data strategy?", dimension="policy",
    )

    assert len(queries) == 4
    assert "France open data strategy" in queries
    client.responses.create.assert_called_once()


def test_generate_queries_core_falls_back_when_too_few_queries_returned():
    client = MagicMock()
    # Only 2 queries returned — below the minimum of 4, should trigger fallback
    client.responses.create.return_value = make_mock_openai_response(
        json.dumps({"queries": ["France open data", "France strategy"]})
    )

    queries = generate_queries_core(
        client=client, country="France", question_id="P2",
        question="Is there a national open data strategy?", dimension="policy",
    )

    # Fallback always returns 5 queries for a policy-dimension question
    assert len(queries) == 5


def test_generate_queries_core_falls_back_on_malformed_json():
    client = MagicMock()
    client.responses.create.return_value = make_mock_openai_response("not valid json at all")

    queries = generate_queries_core(
        client=client, country="Estonia", question_id="I1",
        question="Do you have a definition of open data reuse?", dimension="impact",
    )

    assert len(queries) == 5  # fallback_queries for I1


# ---------------------------------------------------------------------------
# Answer generation (Section 5.3.2)
# ---------------------------------------------------------------------------

def _sample_retrieved_df():
    return pd.DataFrame([
        {
            "retrieval_rank": 1,
            "source_title": "National Open Data Strategy",
            "source_url": "https://gov.example.com/strategy",
            "source_file_type": "pdf",
            "similarity_score": 0.85,
            "chunk_text": "The government has published a national open data strategy.",
        },
        {
            "retrieval_rank": 2,
            "source_title": "Ministry Report",
            "source_url": "https://gov.example.com/report",
            "source_file_type": "html",
            "similarity_score": 0.78,
            "chunk_text": "The strategy was adopted by the ministry in 2023.",
        },
    ])


def test_generate_answer_returns_expected_structure_on_success():
    client = MagicMock()
    client.responses.create.side_effect = [
        make_mock_openai_response(json.dumps({
            "final_answer_selected": "Yes",
            "score_suggestion": 20,
            "justification_generated": "Yes, the country has a national strategy [S1].",
            "confidence": 85,
            "evidence_sufficiency": "sufficient",
        })),
        make_mock_openai_response(json.dumps({"final_answer_selected": "Yes"})),
    ]

    result = generate_answer_for_question(
        country="France", question_id="P2", question_text="Is there a national open data strategy?",
        dimension="policy", response_scoring="{'yes': 20, 'no': 0}",
        retrieved_df=_sample_retrieved_df(), client=client,
    )

    assert result["final_answer_selected"].lower() == "yes"
    assert result["score_suggestion"] == 20
    assert result["evidence_sufficiency"] == "sufficient"
    assert result["retrieved_evidence_count"] == 2
    assert client.responses.create.call_count == 2  # first-stage + second-stage


def test_generate_answer_uses_fallback_when_no_evidence_retrieved():
    client = MagicMock()

    result = generate_answer_for_question(
        country="France", question_id="P2", question_text="Is there a national open data strategy?",
        dimension="policy", response_scoring="{'yes': 20, 'no': 0}",
        retrieved_df=pd.DataFrame(), client=client,
    )

    assert result["evidence_sufficiency"] == "weak"
    assert "too limited" in result["justification_generated"] or "too limited" in result["justification_generated"].lower()
    # No API call should be made for the first-stage generation when there's no evidence,
    # since qdf.empty short-circuits straight to fallback_output
    assert client.responses.create.call_count <= 1  # only the second-stage selector may still be called


def test_generate_answer_falls_back_gracefully_on_api_exception():
    client = MagicMock()
    client.responses.create.side_effect = Exception("simulated API failure")

    result = generate_answer_for_question(
        country="France", question_id="P2", question_text="Is there a national open data strategy?",
        dimension="policy", response_scoring="{'yes': 20, 'no': 0}",
        retrieved_df=_sample_retrieved_df(), client=client,
    )

    assert result["evidence_sufficiency"] == "weak"
    assert result["confidence"] == 10


def test_generate_answer_removes_inline_source_markers_from_justification():
    client = MagicMock()
    client.responses.create.side_effect = [
        make_mock_openai_response(json.dumps({
            "final_answer_selected": "Yes",
            "score_suggestion": 20,
            "justification_generated": "Yes, the strategy exists [S1] and was adopted (S2).",
            "confidence": 80,
            "evidence_sufficiency": "sufficient",
        })),
        make_mock_openai_response(json.dumps({"final_answer_selected": "Yes"})),
    ]

    result = generate_answer_for_question(
        country="France", question_id="P2", question_text="Is there a national open data strategy?",
        dimension="policy", response_scoring="{'yes': 20, 'no': 0}",
        retrieved_df=_sample_retrieved_df(), client=client,
    )

    assert "[S1]" not in result["justification_generated"]
    assert "(S2)" not in result["justification_generated"]
    assert "## Sources" in result["justification_generated"]


def test_generate_answer_second_stage_can_override_first_stage_answer():
    client = MagicMock()
    client.responses.create.side_effect = [
        make_mock_openai_response(json.dumps({
            "final_answer_selected": "No",  # first-stage says No
            "score_suggestion": 0,
            "justification_generated": "No clear strategy found.",
            "confidence": 60,
            "evidence_sufficiency": "partial",
        })),
        make_mock_openai_response(json.dumps({"final_answer_selected": "Yes"})),  # second-stage overrides to Yes
    ]

    result = generate_answer_for_question(
        country="France", question_id="P2", question_text="Is there a national open data strategy?",
        dimension="policy", response_scoring="{'yes': 20, 'no': 0}",
        retrieved_df=_sample_retrieved_df(), client=client,
    )

    assert result["final_answer_selected"].lower() == "yes"
    assert result["score_suggestion"] == 20  # score correctly derived from the FINAL (second-stage) answer