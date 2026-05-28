"""Tests for council orchestration (members + chairman) with mocked generation."""

from unittest.mock import MagicMock

from mt_llm.council.council import Council


def _mock_member(translation: str, review: str = "ok") -> MagicMock:
    m = MagicMock()
    m.translate.return_value = translation
    m.review.return_value = review
    return m


def _mock_chairman(final: str) -> MagicMock:
    c = MagicMock()
    c.arbitrate.return_value = final
    return c


def test_council_runs_members_then_chairman():
    council = Council.__new__(Council)  # bypass __init__
    council.members = [
        _mock_member("Hallo Welt"),
        _mock_member("Guten Tag Welt"),
        _mock_member("Hallo zusammen"),
    ]
    council.chairman = _mock_chairman("Hallo Welt")
    council.peer_review = False
    council.terms_to_chairman = False
    council.allow_chairman_rewrite = True

    result = council.run("Hello world")

    assert result["source_text"] == "Hello world"
    assert result["candidates"] == ["Hallo Welt", "Guten Tag Welt", "Hallo zusammen"]
    assert result["final_translation"] == "Hallo Welt"
    assert result["reviews"] is None


def test_council_passes_peer_reviews_when_enabled():
    council = Council.__new__(Council)
    council.members = [
        _mock_member("A", review="review-from-0"),
        _mock_member("B", review="review-from-1"),
    ]
    council.chairman = _mock_chairman("A")
    council.peer_review = True
    council.terms_to_chairman = False
    council.allow_chairman_rewrite = True

    result = council.run("Hello")

    assert result["reviews"] == ["review-from-0", "review-from-1"]
    chairman_call_kwargs = council.chairman.arbitrate.call_args.kwargs
    assert chairman_call_kwargs["reviews"] == ["review-from-0", "review-from-1"]


def test_council_forwards_terms_when_terms_to_chairman():
    council = Council.__new__(Council)
    council.members = [_mock_member("Foo")]
    council.chairman = _mock_chairman("Foo")
    council.peer_review = False
    council.terms_to_chairman = True
    council.allow_chairman_rewrite = True

    council.run("Hello", terms={"Hello": "Hallo"})

    chairman_call_kwargs = council.chairman.arbitrate.call_args.kwargs
    assert chairman_call_kwargs["terms"] == {"Hello": "Hallo"}


def test_council_withholds_terms_from_chairman_when_disabled():
    council = Council.__new__(Council)
    council.members = [_mock_member("Foo")]
    council.chairman = _mock_chairman("Foo")
    council.peer_review = False
    council.terms_to_chairman = False
    council.allow_chairman_rewrite = True

    council.run("Hello", terms={"Hello": "Hallo"})

    chairman_call_kwargs = council.chairman.arbitrate.call_args.kwargs
    assert chairman_call_kwargs["terms"] is None


def test_council_propagates_allow_rewrite_flag():
    council = Council.__new__(Council)
    council.members = [_mock_member("Foo")]
    council.chairman = _mock_chairman("Foo")
    council.peer_review = False
    council.terms_to_chairman = False
    council.allow_chairman_rewrite = False

    council.run("Hello")
    assert council.chairman.arbitrate.call_args.kwargs["allow_rewrite"] is False
