"""Tests for council prompt templates."""

from mt_llm.council.prompts import (
    build_chairman_prompt,
    build_member_review_prompt,
    build_member_translate_prompt,
)


def test_member_translate_includes_source():
    prompt = build_member_translate_prompt("Hello world")
    assert "Hello world" in prompt
    assert "German" in prompt or "Deutsch" in prompt


def test_member_translate_includes_terms_when_given():
    prompt = build_member_translate_prompt("Hello", terms={"Hello": "Hallo"})
    assert "Hello" in prompt
    assert "Hallo" in prompt


def test_member_translate_no_terms_section_when_empty():
    prompt = build_member_translate_prompt("Hello")
    assert "Required terms" not in prompt


def test_review_prompt_includes_all_candidates():
    prompt = build_member_review_prompt(
        source="Hello",
        own_translation="Hallo",
        other_translations=["Guten Tag", "Grüß Gott"],
    )
    assert "Hallo" in prompt
    assert "Guten Tag" in prompt
    assert "Grüß Gott" in prompt


def test_chairman_prompt_lists_candidates():
    prompt = build_chairman_prompt(
        source="Hello",
        candidates=["Hallo", "Guten Tag"],
        reviews=None,
        terms=None,
        allow_rewrite=True,
    )
    assert "Hallo" in prompt
    assert "Guten Tag" in prompt
    assert "Hello" in prompt


def test_chairman_prompt_includes_terms_when_given():
    prompt = build_chairman_prompt(
        source="Open the consumption model",
        candidates=["Öffnen Sie das Modell"],
        reviews=None,
        terms={"consumption model": "Verbrauchsmodell"},
        allow_rewrite=True,
    )
    assert "Verbrauchsmodell" in prompt


def test_chairman_prompt_includes_reviews_when_given():
    prompt = build_chairman_prompt(
        source="Hello",
        candidates=["Hallo"],
        reviews=["candidate 1 is fluent"],
        terms=None,
        allow_rewrite=True,
    )
    assert "candidate 1 is fluent" in prompt


def test_chairman_prompt_forbids_rewrite_when_disallowed():
    prompt = build_chairman_prompt(
        source="Hello",
        candidates=["Hallo"],
        reviews=None,
        terms=None,
        allow_rewrite=False,
    )
    assert "select" in prompt.lower() or "choose" in prompt.lower()
