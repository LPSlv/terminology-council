"""
Prompt templates for the 2-stage translation pipeline.
"""

import json
import os
from typing import Dict, Optional, Literal, Tuple
from utils import get_terminology_fields

# Cache for the example from test_data.json
_example_cache: Optional[Dict] = None


def _load_example_from_test_data() -> Optional[Dict]:
    """
    Load the first entry from test_data.json with safe fallback.
    
    Returns:
        Dictionary with keys: 'en', 'de', 'proper_terms' (and optionally 'random_terms'),
        or None if file cannot be loaded.
    """
    global _example_cache
    
    if _example_cache is not None:
        return _example_cache
    
    try:
        # Find test_data.json relative to this file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        test_data_path = os.path.join(script_dir, 'test_data.json')
        
        if not os.path.exists(test_data_path):
            _example_cache = None
            return None
        
        with open(test_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list) and len(data) > 0:
            entry = data[0]
            # Extract relevant fields
            proper_terms, _ = get_terminology_fields(entry)
            example = {
                'en': entry.get('en', ''),
                'de': entry.get('de', ''),
                'proper_terms': proper_terms
            }
            _example_cache = example
            return example
        else:
            _example_cache = None
            return None
    except Exception:
        # Safe fallback: return None if anything goes wrong
        _example_cache = None
        return None


def _get_corrected_example_german() -> str:
    """
    Return a manually corrected version of the example German translation.
    
    The original has errors like "un" (should be "und") and "Attribute" (should be "Attributen").
    This demonstrates minimal edits to fix grammar while keeping structure.
    """
    # Original: "Öffnen Sie das Verbrauchsmodell mit den Kennzahlen und Attribute, die Sie in Ihre Perspektive aufnehmen möchten, un wechseln Sie zur Registerkarte Perspektiven."
    # Corrections:
    # - "Attribute" -> "Attributen" (dative plural)
    # - "un" -> "und"
    # - Align instruction with the source "click": "wechseln" -> "klicken Sie auf"
    return (
        "Öffnen Sie das Verbrauchsmodell mit den Kennzahlen und Attributen, die Sie in Ihre Perspektive aufnehmen möchten, "
        "und klicken Sie auf die Registerkarte Perspektiven."
    )


def get_mt_prompt(source_text: str) -> str:
    """
    Prompt for MT models (may be unused for pure MT pipeline, kept minimal).
    
    Args:
        source_text: English text to translate
        
    Returns:
        Prompt string
    """
    return source_text  # Most MT models take text directly


def get_checker_prompt(
    source_text: str,
    mt_output: str,
    terms: Optional[Dict[str, str]] = None,
    memory: Optional[Dict[str, str]] = None,
    terminology_mode: Literal["on", "off"] = "on",
) -> str:
    """
    Prompt for checker/post-editor stage.
    
    Args:
        source_text: Original English text
        mt_output: Machine translation output
        terms: Optional terminology dictionary (English -> German)
        memory: Optional consistency memory from previous runs
        terminology_mode: "on" to enforce provided terminology, "off" to ignore it
        
    Returns:
        Prompt string
    """
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

    # Try to load example from test_data.json
    example_data = _load_example_from_test_data()
    
    # Build example section if available
    example_section = ""
    if example_data:
        example_en = example_data.get('en', '')
        example_de = example_data.get('de', '')
        example_terms = example_data.get('proper_terms', {})
        corrected_example_de = _get_corrected_example_german()
        
        # Build terms section for example (only if terminology_mode is on)
        example_terms_section = ""
        if enforce_terms and example_terms:
            example_terms_list = ", ".join([f"'{en}'='{de}'" for en, de in example_terms.items()])
            example_terms_section = f"\nRequired terms (use these exact translations): {example_terms_list}\n"
        
        example_section = f"""Example:

English: {example_en}
Current German translation: {example_de}{example_terms_section}
Corrected German translation: {corrected_example_de}

---

"""

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
