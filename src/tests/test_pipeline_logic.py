"""
Unit tests for the pure-logic components of the RAG pipeline: chunking,
ground-truth score inference, response canonicalisation, and evaluation
metrics. These functions were selected for testing because they involve
no network calls or external API dependencies, and because they implement
core, verifiable claims made in the report (e.g. chunk size, response
matching, metric correctness).
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from step7_chunk_text import chunk_text_by_words
from step2_build_ground_truth import infer_awarded_score, parse_response_scoring
from step11_evaluate_chroma import (
    canonicalize_answer_label,
    tfidf_cosine_similarity,
    jaccard_similarity,
    bleu_score,
    rouge_l_f1,
)


# ---------------------------------------------------------------------------
# Chunking (Section 5.2.5)
# ---------------------------------------------------------------------------

def test_chunking_respects_configured_chunk_size():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text_by_words(text, chunk_size=180, overlap=40)

    for chunk in chunks[:-1]:  # all but the final chunk should be full-length
        assert len(chunk.split()) == 180


def test_chunking_overlap_between_consecutive_chunks():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text_by_words(text, chunk_size=180, overlap=40)

    first_chunk_words = chunks[0].split()
    second_chunk_words = chunks[1].split()

    overlap_region = first_chunk_words[-40:]
    assert overlap_region == second_chunk_words[:40]


def test_chunking_final_chunk_may_be_shorter():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text_by_words(text, chunk_size=180, overlap=40)

    assert len(chunks[-1].split()) < 180


def test_chunking_empty_text_returns_no_chunks():
    assert chunk_text_by_words("") == []
    assert chunk_text_by_words("   ") == []


# ---------------------------------------------------------------------------
# Ground-truth score inference (Section 5.2)
# ---------------------------------------------------------------------------

def test_infer_awarded_score_matches_case_insensitively():
    scoring = "{'yes': 20, 'no': 0}"
    assert infer_awarded_score("Yes", scoring) == 20
    assert infer_awarded_score("yes", scoring) == 20
    assert infer_awarded_score("YES", scoring) == 20


def test_infer_awarded_score_no_match_returns_none():
    scoring = "{'yes': 20, 'no': 0}"
    assert infer_awarded_score("maybe", scoring) is None


def test_infer_awarded_score_handles_multi_tier_scale():
    scoring = "{'all public bodies': 20, 'few public bodies': 5, 'none of the public bodies': 0}"
    assert infer_awarded_score("Few public bodies", scoring) == 5


# ---------------------------------------------------------------------------
# Response canonicalisation (Section 5.3.2.1 / 6.1.2)
# ---------------------------------------------------------------------------

def test_canonicalize_answer_label_normalizes_case_and_formatting():
    assert canonicalize_answer_label("YES") == "Yes"
    assert canonicalize_answer_label("  yes  ") == "Yes"
    assert canonicalize_answer_label("n/a") == "Not applicable"


def test_canonicalize_answer_label_multi_tier_scale():
    assert canonicalize_answer_label("THE MAJORITY OF PUBLIC BODIES") == "The majority of public bodies"


def test_canonicalize_answer_label_unrecognised_input_passthrough():
    assert canonicalize_answer_label("Some free-text answer") == "Some free-text answer"


def test_canonicalize_answer_label_empty_input():
    assert canonicalize_answer_label("") == ""
    assert canonicalize_answer_label(None) == ""


# ---------------------------------------------------------------------------
# Evaluation metrics (Section 6.1.2)
# ---------------------------------------------------------------------------

def test_tfidf_similarity_identical_text_is_one():
    text = "the system retrieves evidence from official government sources"
    sim = tfidf_cosine_similarity(text, text)
    assert math.isclose(sim, 1.0, abs_tol=1e-6)


def test_tfidf_similarity_unrelated_text_is_low():
    sim = tfidf_cosine_similarity(
        "the system retrieves evidence from government sources",
        "cats and dogs are common household pets",
    )
    assert sim < 0.2


def test_jaccard_similarity_identical_text_is_one():
    text = "official government strategy document"
    assert jaccard_similarity(text, text) == 1.0


def test_jaccard_similarity_partial_overlap():
    sim = jaccard_similarity("the national data strategy", "the national open data policy")
    assert 0.0 < sim < 1.0


def test_bleu_score_identical_text_is_high():
    text = "the country has published a national open data strategy"
    score = bleu_score(text, text)
    assert score > 0.9


def test_rouge_l_f1_identical_text_is_one():
    text = "the country has published a national open data strategy"
    score = rouge_l_f1(text, text)
    assert math.isclose(score, 1.0, abs_tol=1e-6)


def test_rouge_l_f1_unrelated_text_is_low():
    score = rouge_l_f1(
        "the country has published a national open data strategy",
        "unrelated sentence about something else entirely",
    )
    assert score < 0.3