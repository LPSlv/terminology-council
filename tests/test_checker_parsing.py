"""Tests for the JSON extraction logic that parses checker LLM output."""

import pytest

from mt_llm.io import parse_json_with_retry

GOOD = (
    '{"final_translation": "Hallo Welt", '
    '"changes": [], "terminology_issues": [], "consistency_notes": []}'
)


def test_parses_clean_json():
    out = parse_json_with_retry(GOOD, attempt=0, max_retries=2, original_prompt="")
    assert out["final_translation"] == "Hallo Welt"
    assert out["changes"] == []


def test_parses_json_in_markdown_code_block():
    payload = f"Sure! Here you go:\n```json\n{GOOD}\n```\nDone."
    out = parse_json_with_retry(payload, attempt=0, max_retries=2, original_prompt="")
    assert out["final_translation"] == "Hallo Welt"


def test_normalises_lists_when_wrong_type():
    payload = (
        '{"final_translation": "Hallo", '
        '"changes": "not a list", '
        '"terminology_issues": null, '
        '"consistency_notes": 42}'
    )
    out = parse_json_with_retry(payload, attempt=0, max_retries=2, original_prompt="")
    assert out["changes"] == []
    assert out["terminology_issues"] == []
    assert out["consistency_notes"] == []


def test_missing_required_key_raises():
    payload = '{"changes": [], "terminology_issues": [], "consistency_notes": []}'
    with pytest.raises(ValueError, match="Missing required key"):
        parse_json_with_retry(payload, attempt=0, max_retries=2, original_prompt="")


def test_garbage_with_no_json_raises():
    with pytest.raises(ValueError):
        parse_json_with_retry("there is no json here", attempt=0, max_retries=2, original_prompt="")
