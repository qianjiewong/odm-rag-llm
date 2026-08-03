"""
Unit tests for the retrieval and reranking logic in step9_retrieve_topk_chroma.py:
benchmark-leakage blocking, official-source weighting, question-aware reranking,
low-value penalties, the Belgium-targeted refinement, and source diversification.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from step9_retrieve_topk_chroma import (
    is_blocked_source,
    is_official_domain,
    get_official_source_bonus,
    get_low_value_penalty,
    get_question_rerank_bonus,
    get_belgium_targeted_bonus,
    get_belgium_targeted_penalty,
    get_dynamic_max_chunks_per_source,
    diversify_candidates,
    get_domain,
)


# ---------------------------------------------------------------------------
# Benchmark-leakage blocking (Section 5.2.3)
# ---------------------------------------------------------------------------

def test_blocks_odm_factsheet_url():
    assert is_blocked_source("https://data.europa.eu/country-factsheet-2025.pdf", "Some title") is True


def test_blocks_questionnaire_title():
    assert is_blocked_source("https://example.com/page", "2025 Open Data Maturity Questionnaire") is True


def test_allows_legitimate_government_source():
    assert is_blocked_source("https://www.gov.uk/open-data-strategy", "National Open Data Strategy") is False


# ---------------------------------------------------------------------------
# Official-source weighting (Section 5.3.1.2)
# ---------------------------------------------------------------------------

def test_official_domain_detection():
    assert is_official_domain("www.gov.uk") is True
    assert is_official_domain("ec.europa.eu") is True
    assert is_official_domain("www.randomblog.com") is False


def test_official_source_bonus_stacks_correctly():
    # Official domain + ministry keyword + PDF = all three components should apply
    bonus = get_official_source_bonus(
        source_url="https://www.gov.uk/report.pdf",
        source_title="Ministry of Digital Government Strategy",
        source_file_type="pdf",
    )
    assert bonus == 0.08 + 0.04 + 0.02


def test_official_source_bonus_zero_for_unofficial_generic_source():
    bonus = get_official_source_bonus(
        source_url="https://randomblog.com/post",
        source_title="My thoughts on data",
        source_file_type="html",
    )
    assert bonus == 0.0


# ---------------------------------------------------------------------------
# Low-value-source penalties (Section 5.3.1.5)
# ---------------------------------------------------------------------------

def test_low_value_penalty_short_chunk():
    penalty = get_low_value_penalty(
        source_title="Some Report",
        source_url="https://example.com/report",
        chunk_text=" ".join(["word"] * 20),  # under 40 words
    )
    assert penalty == 0.03


def test_low_value_penalty_generic_markers():
    penalty = get_low_value_penalty(
        source_title="Homepage - Welcome",
        source_url="https://example.com/portal",
        chunk_text="Welcome to our landing page. " * 10,
    )
    assert penalty >= 0.05


def test_low_value_penalty_none_for_substantive_content():
    penalty = get_low_value_penalty(
        source_title="National Open Data Strategy 2025",
        source_url="https://gov.example.com/strategy",
        chunk_text=" ".join(["substantive detailed policy content"] * 20),
    )
    assert penalty == 0.0


# ---------------------------------------------------------------------------
# Question-aware reranking (Section 5.3.1.3)
# ---------------------------------------------------------------------------

def test_policy_question_rewards_legislative_language():
    bonus = get_question_rerank_bonus(
        question_id="P1",
        dimension="policy",
        source_title="National Open Data Act",
        source_url="https://example.com",
        chunk_text="This law establishes the national open data strategy.",
        source_file_type="pdf",
    )
    assert bonus > 0.0


def test_regional_question_rewards_local_authority_language():
    bonus = get_question_rerank_bonus(
        question_id="P3",
        dimension="policy",
        source_title="Municipal Open Data Initiative",
        source_url="https://example.com",
        chunk_text="The local authority launched a regional open data programme.",
        source_file_type="html",
    )
    assert bonus >= 0.18


def test_consultation_question_rewards_stakeholder_language():
    bonus = get_question_rerank_bonus(
        question_id="I9",
        dimension="impact",
        source_title="Stakeholder Consultation Report",
        source_url="https://example.com",
        chunk_text="A public consultation workshop was held to engage stakeholders.",
        source_file_type="pdf",
    )
    assert bonus >= 0.16


def test_reranking_bonus_zero_for_irrelevant_evidence():
    bonus = get_question_rerank_bonus(
        question_id="I9",
        dimension="impact",
        source_title="Weather Forecast",
        source_url="https://example.com",
        chunk_text="Tomorrow will be sunny with a light breeze.",
        source_file_type="html",
    )
    assert bonus == 0.0


def test_reranking_bonus_unrecognised_question_id_returns_zero_or_dimension_only():
    # Question ID not in any category set: only the dimension-level bonus can apply
    bonus = get_question_rerank_bonus(
        question_id="X99",
        dimension="policy",
        source_title="National Strategy",
        source_url="https://example.com",
        chunk_text="This policy establishes a national strategy.",
        source_file_type="pdf",
    )
    assert bonus == 0.03  # only the generic policy-dimension bonus applies


# ---------------------------------------------------------------------------
# Belgium-targeted refinement (Section 5.3.1.6) — country gating
# ---------------------------------------------------------------------------

def test_belgium_bonus_zero_for_non_belgium_country():
    bonus = get_belgium_targeted_bonus(
        country="France",
        question_id="I10",
        source_title="Reuse case catalogue",
        source_url="https://example.com",
        chunk_text="This is an example of reuse case showcase.",
        source_file_type="html",
    )
    assert bonus == 0.0


def test_belgium_bonus_applies_for_belgium_matching_question():
    bonus = get_belgium_targeted_bonus(
        country="Belgium",
        question_id="I10",
        source_title="Reuse case catalogue",
        source_url="https://example.com",
        chunk_text="This is an example of a reuse case showcase.",
        source_file_type="html",
    )
    assert bonus > 0.0


def test_belgium_bonus_zero_for_belgium_but_uncovered_question():
    # Belgium-gated, but question ID not one of the five targeted questions
    bonus = get_belgium_targeted_bonus(
        country="Belgium",
        question_id="P13",
        source_title="Some governance document",
        source_url="https://example.com",
        chunk_text="A governance structure was established.",
        source_file_type="pdf",
    )
    assert bonus == 0.0


def test_belgium_penalty_zero_for_non_belgium_country():
    penalty = get_belgium_targeted_penalty(
        country="Bulgaria",
        question_id="P24",
        source_title="Legal framework overview",
        source_url="https://example.com",
        chunk_text="This describes the legal framework in general terms.",
    )
    assert penalty == 0.0


def test_belgium_penalty_applies_for_generic_evidence_without_signal():
    penalty = get_belgium_targeted_penalty(
        country="Belgium",
        question_id="P24",
        source_title="Legal framework overview",
        source_url="https://example.com",
        chunk_text="This describes the legal framework in general terms.",
    )
    assert penalty > 0.0


def test_belgium_penalty_zero_when_signal_present():
    penalty = get_belgium_targeted_penalty(
        country="Belgium",
        question_id="P24",
        source_title="Policy review and approval status",
        source_url="https://example.com",
        chunk_text="The strategy was recently revised and is pending approval.",
    )
    assert penalty == 0.0


# ---------------------------------------------------------------------------
# Source diversification (Section 5.3.1.4)
# ---------------------------------------------------------------------------

def _make_candidate(chunk_id, source_title, source_url):
    return {
        "chunk_id": chunk_id,
        "source_title": source_title,
        "source_url": source_url,
    }


def test_diversify_selects_only_one_chunk_per_source_in_first_pass():
    candidates = [
        _make_candidate("c1", "Source A", "https://a.com"),
        _make_candidate("c2", "Source A", "https://a.com"),  # same source as above
        _make_candidate("c3", "Source B", "https://b.com"),
    ]
    # top_k=2 forces the function to stop within the first (one-per-source) pass,
    # before backfill logic is reached
    selected = diversify_candidates(candidates, question_id="P13", top_k=2)

    assert len(selected) == 2
    assert selected[0]["chunk_id"] == "c1"
    assert selected[1]["chunk_id"] == "c3"
    source_keys = [(row["source_title"], row["source_url"]) for row in selected]
    assert len(source_keys) == len(set(source_keys))


def test_diversify_respects_dynamic_max_chunks_for_single_chunk_questions():
    # P1 is in the "max 1 chunk per source" set
    candidates = [
        _make_candidate("c1", "Source A", "https://a.com"),
        _make_candidate("c2", "Source A", "https://a.com"),
        _make_candidate("c3", "Source A", "https://a.com"),
    ]
    selected = diversify_candidates(candidates, question_id="P1", top_k=8)
    assert len(selected) == 1


def test_diversify_allows_two_chunks_per_source_for_default_questions():
    # P13 is NOT in the restricted set, so default max (2) applies
    candidates = [
        _make_candidate("c1", "Source A", "https://a.com"),
        _make_candidate("c2", "Source A", "https://a.com"),
        _make_candidate("c3", "Source A", "https://a.com"),
    ]
    selected = diversify_candidates(candidates, question_id="P13", top_k=8)
    assert len(selected) == 2


def test_dynamic_max_chunks_per_source_values():
    assert get_dynamic_max_chunks_per_source("P1") == 1
    assert get_dynamic_max_chunks_per_source("P13") == 2
    assert get_dynamic_max_chunks_per_source("I27") == 1


def test_diversify_backfill_adds_second_chunk_when_top_k_not_reached():
    # With top_k larger than the number of unique sources, backfill should
    # add a second chunk from Source A (P13 allows up to 2 per source)
    candidates = [
        _make_candidate("c1", "Source A", "https://a.com"),
        _make_candidate("c2", "Source A", "https://a.com"),
        _make_candidate("c3", "Source B", "https://b.com"),
    ]
    selected = diversify_candidates(candidates, question_id="P13", top_k=8)

    assert len(selected) == 3
    source_a_count = sum(1 for row in selected if row["source_title"] == "Source A")
    assert source_a_count == 2


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def test_get_domain_extracts_netloc():
    assert get_domain("https://www.gov.uk/strategy") == "www.gov.uk"


def test_get_domain_handles_malformed_url_gracefully():
    assert get_domain("not a url at all") == ""