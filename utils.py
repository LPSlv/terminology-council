"""
Utility functions for the 2-stage translation pipeline.
"""

import json
import re
from typing import Dict, Optional, List, Tuple


def parse_json_with_retry(
    output: str,
    attempt: int,
    max_retries: int,
    original_prompt: str
) -> Dict:
    """
    Parse JSON from checker output with retry logic.
    
    Args:
        output: Raw output from checker
        attempt: Current attempt number
        max_retries: Maximum number of retries
        original_prompt: Original prompt (for error messages)
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        json.JSONDecodeError: If JSON cannot be parsed
        ValueError: If JSON structure is invalid
    """
    # Clean the text first - remove markdown code blocks if present
    cleaned_text = output.strip()
    
    # Remove markdown code blocks
    if "```" in cleaned_text:
        # Extract JSON from code blocks
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned_text, re.DOTALL)
        if code_block_match:
            cleaned_text = code_block_match.group(1)
    
    # Try to find JSON object (use non-greedy to get first complete object)
    json_match = re.search(r'\{.*?\}', cleaned_text, re.DOTALL)
    if not json_match:
        # Fallback: try greedy match for nested objects
        json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
    
    if json_match:
        json_str = json_match.group()
        # Try to fix common JSON issues
        json_str = json_str.replace("'", '"')  # Replace single quotes with double
        try:
            data = json.loads(json_str)
            # Validate structure
            required_keys = ["final_translation", "changes", "terminology_issues", "consistency_notes"]
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing required key: {key}")
            
            # Ensure lists are lists
            for key in ["changes", "terminology_issues", "consistency_notes"]:
                if not isinstance(data[key], list):
                    data[key] = []
            
            return data
        except json.JSONDecodeError as e:
            # Re-raise with more context
            raise ValueError(f"Invalid JSON: {e}")
    
    raise ValueError(f"No valid JSON found in output: {cleaned_text[:200]}")


def match_terminology(
    translation: str,
    terms: Dict[str, str]
) -> List[Dict[str, str]]:
    """
    Simple terminology matching helper.
    
    Args:
        translation: German translation text
        terms: Dictionary mapping English terms to expected German terms
        
    Returns:
        List of terminology issues found
    """
    issues = []
    translation_lower = translation.lower()
    
    for en_term, expected_de_term in terms.items():
        # Simple string matching (case-insensitive)
        pattern = r'\b' + re.escape(expected_de_term.lower()) + r'\b'
        if not re.search(pattern, translation_lower):
            # Check if a similar but incorrect term appears
            found = None
            # Simple heuristic: look for words that might be related
            words = translation_lower.split()
            for word in words:
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
    """
    Extract terminology fields from a dataset entry, handling both old and new formats.
    
    Handles both formats:
    - Old format: 'proper_terms', 'random_terms'
    - New format: 'proper', 'random'
    
    Args:
        entry: Dictionary entry from JSONL file
        
    Returns:
        Tuple of (proper_terms_dict, random_terms_dict_or_None)
    """
    # Try new format first (proper, random), then fall back to old format (proper_terms, random_terms)
    proper_terms = entry.get("proper", entry.get("proper_terms", {}))
    random_terms = entry.get("random", entry.get("random_terms", None))
    
    return proper_terms, random_terms


def load_terms_from_json(file_path: str) -> Dict[str, str]:
    """
    Load terminology dictionary from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary mapping English terms to German terms
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(data, dict):
                # Direct mapping
                if all(isinstance(v, str) for v in data.values()):
                    return data
                # Nested structure (e.g., {"proper_terms": {...}} or {"proper": {...}})
                if "proper_terms" in data:
                    return data["proper_terms"]
                if "proper" in data:
                    return data["proper"]
            elif isinstance(data, list) and len(data) > 0:
                # List of entries
                # Collect terms from all entries
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
