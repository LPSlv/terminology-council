"""JSONL I/O and terminology helpers."""

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_json_with_retry(
    output: str,
    attempt: int,
    max_retries: int,
    original_prompt: str
) -> Dict:
    """Extract and validate a JSON object from raw checker output."""
    cleaned_text = output.strip()

    if "```" in cleaned_text:
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned_text, re.DOTALL)
        if code_block_match:
            cleaned_text = code_block_match.group(1)

    # Non-greedy first to get the outermost complete object
    json_match = re.search(r'\{.*?\}', cleaned_text, re.DOTALL)
    if not json_match:
        json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)

    if json_match:
        json_str = json_match.group()
        json_str = json_str.replace("'", '"')
        try:
            data = json.loads(json_str)
            required_keys = ["final_translation", "changes", "terminology_issues", "consistency_notes"]
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing required key: {key}")
            for key in ["changes", "terminology_issues", "consistency_notes"]:
                if not isinstance(data[key], list):
                    data[key] = []
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

    raise ValueError(f"No valid JSON found in output: {cleaned_text[:200]}")


def match_terminology(
    translation: str,
    terms: Dict[str, str]
) -> List[Dict[str, str]]:
    """Return issues for any expected German terms missing from the translation."""
    issues = []
    translation_lower = translation.lower()

    for en_term, expected_de_term in terms.items():
        pattern = r'\b' + re.escape(expected_de_term.lower()) + r'\b'
        if not re.search(pattern, translation_lower):
            # Prefix heuristic: surface the nearest candidate for easier debugging
            found = None
            for word in translation_lower.split():
                if len(word) > 3 and word[:3] == expected_de_term.lower()[:3]:
                    found = word
                    break
            issues.append({
                "source_term": en_term,
                "expected": expected_de_term,
                "found": found if found else "not found"
            })

    return issues


def get_terminology_fields(entry: Dict) -> Tuple[Dict[str, str], Optional[Dict[str, str]]]:
    """Extract proper/random term dicts from an entry, handling old and new field names."""
    # New format: 'proper'/'random'; old format: 'proper_terms'/'random_terms'
    proper_terms = entry.get("proper", entry.get("proper_terms", {}))
    random_terms = entry.get("random", entry.get("random_terms", None))
    return proper_terms, random_terms


def load_terms_from_json(file_path: str) -> Dict[str, str]:
    """Load a flat English→German terminology dict from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            if all(isinstance(v, str) for v in data.values()):
                return data
            if "proper_terms" in data:
                return data["proper_terms"]
            if "proper" in data:
                return data["proper"]
        elif isinstance(data, list) and len(data) > 0:
            terms = {}
            for entry in data:
                if isinstance(entry, dict):
                    proper_terms, _ = get_terminology_fields(entry)
                    if proper_terms:
                        terms.update(proper_terms)
            return terms

        return {}
    except FileNotFoundError:
        print(f"Warning: Terms file not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in terms file: {e}")
        return {}
    except Exception as e:
        print(f"Warning: Error loading terms file: {e}")
        return {}


def load_jsonl(file_path: str) -> list:
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(file_path: str, rows: Iterable[Dict]) -> None:
    """Write an iterable of dicts to a JSONL file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
