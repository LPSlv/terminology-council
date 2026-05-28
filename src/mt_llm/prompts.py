"""Prompt templates for the 2-stage translation pipeline."""

from typing import Dict, Literal, Optional

# Hard-coded few-shot example for the checker prompt.
# Derived from the first dataset entry; German manually corrected:
# (Attribute -> Attributen, un -> und, structural alignment with "click").
_EXAMPLE_EN = (
    "Open the consumption model containing the measures and attributes "
    "you want to include in your perspective, and click the Perspectives tab."
)
_EXAMPLE_DE_RAW = (
    "Öffnen Sie das Verbrauchsmodell mit den Kennzahlen und Attribute, "
    "die Sie in Ihre Perspektive aufnehmen möchten, un wechseln Sie zur Registerkarte Perspektiven."
)
_EXAMPLE_DE_CORRECTED = (
    "Öffnen Sie das Verbrauchsmodell mit den Kennzahlen und Attributen, "
    "die Sie in Ihre Perspektive aufnehmen möchten, und klicken Sie auf die Registerkarte Perspektiven."
)
_EXAMPLE_TERMS = {"consumption model": "Verbrauchsmodell"}


def get_mt_prompt(source_text: str) -> str:
    return source_text  # Most MT models take text directly


def get_checker_prompt(
    source_text: str,
    mt_output: str,
    terms: Optional[Dict[str, str]] = None,
    memory: Optional[Dict[str, str]] = None,
    terminology_mode: Literal["on", "off"] = "on",
) -> str:
    """Build the post-editor prompt for the checker stage."""
    enforce_terms = (terminology_mode == "on")

    # Combine terms and memory (only when enforcing terminology)
    all_terms: Dict[str, str] = {}
    if enforce_terms:
        if terms:
            all_terms.update(terms)
        if memory:
            all_terms.update(memory)

    terms_section = ""
    if enforce_terms and all_terms:
        terms_list = ", ".join([f"'{en}'='{de}'" for en, de in all_terms.items()])
        terms_section = f"\nRequired terms (use these exact translations): {terms_list}\n"

    first_line = (
        "Review the German translation and make minimal edits to fix terminology and necessary grammatical agreement. Keep the sentence structure unchanged."
        if enforce_terms
        else "Review the German translation and make minimal edits to fix grammar/fluency issues. Keep the sentence structure unchanged."
    )

    rules_line_1 = (
        "- Make minimal edits only: change terms to match the dictionary and fix necessary grammatical agreement"
        if enforce_terms
        else "- Make minimal edits only: fix necessary grammar/fluency issues"
    )

    example_terms_section = ""
    if enforce_terms and _EXAMPLE_TERMS:
        example_terms_list = ", ".join(f"'{en}'='{de}'" for en, de in _EXAMPLE_TERMS.items())
        example_terms_section = f"\nRequired terms (use these exact translations): {example_terms_list}\n"

    example_section = (
        f"Example:\n\n"
        f"English: {_EXAMPLE_EN}\n"
        f"Current German translation: {_EXAMPLE_DE_RAW}"
        f"{example_terms_section}\n"
        f"Corrected German translation: {_EXAMPLE_DE_CORRECTED}\n\n"
        f"---\n\n"
    )

    return f"""{first_line}

Rules:
{rules_line_1}
- Keep sentence structure, word order, and phrasing as much as possible
- Do NOT rewrite, paraphrase, shorten, or change the meaning
- If you cannot improve without changing meaning, output the MT translation unchanged
- Output ONLY the corrected German text, nothing else

{example_section}Task:

English: {source_text}
Current German translation: {mt_output}{terms_section}
Corrected German translation:"""
