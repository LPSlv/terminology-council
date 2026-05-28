"""Tests for JSONL load/write and terminology field extraction."""

import json

from mt_llm.io import (
    get_terminology_fields,
    load_jsonl,
    load_terms_from_json,
    write_jsonl,
)


def test_jsonl_roundtrip(tmp_path):
    rows = [{"en": "Hello", "de": "Hallo"}, {"en": "World", "de": "Welt"}]
    p = tmp_path / "out.jsonl"
    write_jsonl(str(p), rows)
    loaded = load_jsonl(str(p))
    assert loaded == rows


def test_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "in.jsonl"
    p.write_text('{"a": 1}\n\n   \n{"b": 2}\n', encoding="utf-8")
    assert load_jsonl(str(p)) == [{"a": 1}, {"b": 2}]


def test_jsonl_unicode_roundtrip(tmp_path):
    rows = [{"de": "Öffnen Sie die Maßnahme"}]
    p = tmp_path / "u.jsonl"
    write_jsonl(str(p), rows)
    assert load_jsonl(str(p)) == rows


def test_get_terminology_fields_proper_terms_format():
    entry = {"en": "Hi", "de": "Hi", "proper_terms": {"Hi": "Hi"}}
    proper, random = get_terminology_fields(entry)
    assert proper == {"Hi": "Hi"}
    assert random is None


def test_get_terminology_fields_proper_format():
    entry = {"en": "Hi", "de": "Hi", "proper": {"Hi": "Hi"}, "random": {"foo": "bar"}}
    proper, random = get_terminology_fields(entry)
    assert proper == {"Hi": "Hi"}
    assert random == {"foo": "bar"}


def test_get_terminology_fields_missing_returns_empty():
    entry = {"en": "Hi", "de": "Hi"}
    proper, random = get_terminology_fields(entry)
    assert proper == {}
    assert random is None


def test_load_terms_from_json_flat(tmp_path):
    p = tmp_path / "terms.json"
    p.write_text(json.dumps({"Hello": "Hallo", "World": "Welt"}), encoding="utf-8")
    assert load_terms_from_json(str(p)) == {"Hello": "Hallo", "World": "Welt"}


def test_load_terms_from_json_nested(tmp_path):
    p = tmp_path / "terms.json"
    p.write_text(json.dumps({"proper_terms": {"Hello": "Hallo"}}), encoding="utf-8")
    assert load_terms_from_json(str(p)) == {"Hello": "Hallo"}


def test_load_terms_from_json_nested_proper_key(tmp_path):
    p = tmp_path / "terms.json"
    p.write_text(json.dumps({"proper": {"Hello": "Hallo"}}), encoding="utf-8")
    assert load_terms_from_json(str(p)) == {"Hello": "Hallo"}


def test_load_terms_from_json_missing_file_returns_empty(tmp_path):
    assert load_terms_from_json(str(tmp_path / "nope.json")) == {}
