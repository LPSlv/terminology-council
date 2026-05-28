"""Tests for chrF++ and terminology accuracy metrics."""

import pytest

from mt_llm.eval import (
    calculate_chrfpp,
    evaluate_translation,
    terminology_success_rate,
)


def test_chrfpp_identical_strings_is_100():
    score = calculate_chrfpp("Hallo Welt", "Hallo Welt")
    assert score == pytest.approx(100.0, abs=0.01)


def test_chrfpp_completely_different_is_low():
    score = calculate_chrfpp("Hallo Welt", "xxxxxxxxxx")
    assert score < 10.0


def test_chrfpp_partial_match_is_intermediate():
    score = calculate_chrfpp("Hallo Welt", "Hallo Erde")
    assert 20.0 < score < 80.0


def test_term_accuracy_all_found():
    rate, results = terminology_success_rate(
        "Das Verbrauchsmodell enthält Attribute.",
        {"consumption model": "Verbrauchsmodell", "attributes": "Attribute"},
    )
    assert rate == 1.0
    assert all(r["found"] for r in results.values())


def test_term_accuracy_none_found():
    rate, results = terminology_success_rate(
        "Das Modell ist gut.",
        {"consumption model": "Verbrauchsmodell"},
    )
    assert rate == 0.0
    assert not any(r["found"] for r in results.values())


def test_term_accuracy_partial():
    rate, _ = terminology_success_rate(
        "Das Verbrauchsmodell ist gut.",
        {"consumption model": "Verbrauchsmodell", "attributes": "Attribute"},
    )
    assert rate == 0.5


def test_term_accuracy_case_insensitive_by_default():
    rate, _ = terminology_success_rate(
        "Das VERBRAUCHSMODELL ist gut.",
        {"consumption model": "Verbrauchsmodell"},
    )
    assert rate == 1.0


def test_term_accuracy_empty_terms_dict_returns_1():
    rate, results = terminology_success_rate("anything", {})
    assert rate == 1.0
    assert results == {}


def test_evaluate_translation_returns_chrf_and_terminology():
    out = evaluate_translation(
        reference="Das Verbrauchsmodell.",
        hypothesis="Das Verbrauchsmodell.",
        proper_terms={"consumption model": "Verbrauchsmodell"},
    )
    assert "chrfpp_score" in out
    assert out["chrfpp_score"] == pytest.approx(100.0, abs=0.01)
    assert out["terminology"]["proper_terms"]["success_rate"] == 1.0
    assert out["terminology"]["proper_terms"]["total_terms"] == 1
    assert out["terminology"]["proper_terms"]["found_terms"] == 1
