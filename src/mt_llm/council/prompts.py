"""Prompt templates for the LLM Council."""

from typing import Dict, List, Optional


def _format_terms(terms: Optional[Dict[str, str]]) -> str:
    if not terms:
        return ""
    items = ", ".join(f"'{en}'='{de}'" for en, de in terms.items())
    return f"\nRequired terms (use these exact German translations): {items}\n"


def build_member_translate_prompt(
    source: str,
    terms: Optional[Dict[str, str]] = None,
) -> str:
    """Prompt a council member to translate EN -> DE."""
    terms_section = _format_terms(terms)
    return (
        f"Translate the following English text into fluent German.{terms_section}\n"
        f"English: {source}\n"
        f"German:"
    )


def build_member_review_prompt(
    source: str,
    own_translation: str,
    other_translations: List[str],
) -> str:
    """Ask a council member to peer-review the other members' translations."""
    others_block = "\n".join(
        f"  Candidate {i + 1}: {t}" for i, t in enumerate(other_translations)
    )
    return (
        f"You translated this English text into German:\n"
        f"  English: {source}\n"
        f"  Your translation: {own_translation}\n\n"
        f"Other council members produced:\n{others_block}\n\n"
        f"In 1-2 sentences, give a concise critical review of the other candidates: "
        f"which is best, what (if anything) is wrong with each.\n"
        f"Review:"
    )


def build_chairman_prompt(
    source: str,
    candidates: List[str],
    reviews: Optional[List[str]],
    terms: Optional[Dict[str, str]],
    allow_rewrite: bool,
) -> str:
    """Final-arbiter prompt for the chairman."""
    candidates_block = "\n".join(
        f"  Candidate {i + 1}: {c}" for i, c in enumerate(candidates)
    )
    reviews_block = ""
    if reviews:
        reviews_block = "\nPeer reviews:\n" + "\n".join(
            f"  Review {i + 1}: {r}" for i, r in enumerate(reviews)
        ) + "\n"
    terms_section = _format_terms(terms)

    instruction = (
        "Select the best candidate and, if necessary, edit it to fix any remaining "
        "terminology or grammatical issues."
        if allow_rewrite
        else "Select the single best candidate as-is."
    )

    return (
        f"You are the chairman of a translation council. The English source is:\n"
        f"  {source}\n\n"
        f"Council members proposed:\n{candidates_block}\n"
        f"{reviews_block}{terms_section}\n"
        f"{instruction}\n"
        f"Output only the final German translation, nothing else.\n"
        f"Final German translation:"
    )
