"""
Evaluation metrics for the LLM Council Translation System.

Implements:
1. Overall Translation Quality (chrF++)
2. Terminology Success Rate
"""

from typing import Dict, List, Tuple, Optional
from collections import Counter
import re


def calculate_chrfpp(reference: str, hypothesis: str) -> float:
    """
    Calculate chrF++ score between reference and hypothesis translations.
    Uses sacrebleu for accurate chrF++ calculation.
    
    Args:
        reference: Reference translation (ground truth)
        hypothesis: System translation (hypothesis)
    
    Returns:
        chrF++ score (0.0 to 100.0)
    """
    try:
        import sacrebleu
        # chrF++ with word order (default: word_order=2)
        chrf = sacrebleu.sentence_chrf(hypothesis, [reference], word_order=2)
        return chrf.score  # chrF++ score is already in 0-100 scale
    except ImportError:
        raise ImportError("sacrebleu is required for chrF++ calculation. Install with: pip install sacrebleu")


def terminology_success_rate(
    hypothesis: str,
    proper_terms: Dict[str, str],
    case_sensitive: bool = False
) -> Tuple[float, Dict[str, bool]]:
    """
    Calculate terminology success rate by checking if correct term translations
    appear in the hypothesis.
    
    Args:
        hypothesis: System translation
        proper_terms: Dictionary mapping source terms to correct target translations
        case_sensitive: Whether to perform case-sensitive matching
    
    Returns:
        Tuple of (success_rate, term_results) where term_results maps
        source_term -> (found, expected, actual)
    """
    if not proper_terms:
        return 1.0, {}
    
    term_results = {}
    found_count = 0
    
    text = hypothesis if case_sensitive else hypothesis.lower()
    
    for source_term, expected_translation in proper_terms.items():
        expected = expected_translation if case_sensitive else expected_translation.lower()
        
        # Check if expected translation appears in hypothesis
        # Use word boundaries for better matching
        pattern = r'\b' + re.escape(expected) + r'\b'
        found = bool(re.search(pattern, text, re.IGNORECASE if not case_sensitive else 0))
        
        term_results[source_term] = {
            "found": found,
            "expected": expected_translation,
            "source": source_term
        }
        
        if found:
            found_count += 1
    
    success_rate = found_count / len(proper_terms) if proper_terms else 1.0
    return success_rate, term_results


def terminology_consistency(
    hypotheses: List[str],
    source_term: str,
    expected_translation: Optional[str] = None
) -> Tuple[float, Dict[str, int]]:
    """
    Calculate terminology consistency across multiple translations.
    Measures the percentage of deviations from the most consistent translation.
    
    Args:
        hypotheses: List of translations (can be same sentence translated multiple times,
                   or different sentences containing the same term)
        source_term: The source term to check consistency for
        expected_translation: Optional expected translation (if known)
    
    Returns:
        Tuple of (consistency_score, translation_counts) where:
        - consistency_score: Percentage of translations using the most common translation
        - translation_counts: Dictionary mapping translations to their occurrence counts
    """
    if not hypotheses:
        return 0.0, {}
    
    # Extract all possible translations of the source term from hypotheses
    # This is simplified - in practice, you'd need alignment or term extraction
    translations = []
    
    for hyp in hypotheses:
        # Simple approach: look for the expected translation if provided
        if expected_translation:
            pattern = r'\b' + re.escape(expected_translation) + r'\b'
            if re.search(pattern, hyp, re.IGNORECASE):
                translations.append(expected_translation)
            else:
                # Try to find similar terms (simplified)
                # In practice, you'd use proper term alignment
                translations.append(None)  # Mark as inconsistent
        else:
            # Without expected translation, we can't determine consistency
            # This would require term alignment tools
            translations.append(None)
    
    # Count occurrences of each translation
    translation_counts = Counter(translations)
    
    if not translation_counts:
        return 0.0, {}
    
    # Find most common translation
    most_common_translation, most_common_count = translation_counts.most_common(1)[0]
    
    # Calculate consistency: percentage using most common translation
    total = len(translations)
    consistency_score = most_common_count / total if total > 0 else 0.0
    
    return consistency_score, dict(translation_counts)


def evaluate_translation(
    reference: str,
    hypothesis: str,
    proper_terms: Dict[str, str],
    random_terms: Optional[Dict[str, str]] = None
) -> Dict:
    """
    Comprehensive evaluation of a single translation.
    
    Args:
        reference: Reference translation
        hypothesis: System translation
        proper_terms: Dictionary of technical terms (source -> target)
        random_terms: Optional dictionary of common terms for comparison (ignored)
    
    Returns:
        Dictionary containing evaluation metrics (chrF++ and terminology accuracy)
    """
    results = {
        "chrfpp_score": calculate_chrfpp(reference, hypothesis),
        "terminology": {}
    }
    
    # Terminology Success Rate for proper terms
    proper_success_rate, proper_term_results = terminology_success_rate(
        hypothesis, proper_terms
    )
    results["terminology"]["proper_terms"] = {
        "success_rate": proper_success_rate,
        "term_results": proper_term_results,
        "total_terms": len(proper_terms),
        "found_terms": sum(1 for r in proper_term_results.values() if r["found"])
    }
    
    return results


def evaluate_consistency_across_translations(
    hypotheses: List[str],
    term_dictionary: Dict[str, str]
) -> Dict[str, Dict]:
    """
    Evaluate terminology consistency across multiple translations.
    
    Args:
        hypotheses: List of translations (same or different sentences)
        term_dictionary: Dictionary mapping source terms to expected translations
    
    Returns:
        Dictionary mapping source_term -> consistency metrics
    """
    consistency_results = {}
    
    for source_term, expected_translation in term_dictionary.items():
        consistency_score, translation_counts = terminology_consistency(
            hypotheses, source_term, expected_translation
        )
        
        consistency_results[source_term] = {
            "consistency_score": consistency_score,
            "expected": expected_translation,
            "translation_variants": translation_counts,
            "total_occurrences": sum(translation_counts.values())
        }
    
    return consistency_results


def format_evaluation_report(results: Dict) -> str:
    """
    Format evaluation results as a readable report.
    
    Args:
        results: Evaluation results dictionary
    
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 80)
    report.append("EVALUATION REPORT")
    report.append("=" * 80)
    
    # chrF++ Score
    report.append(f"\nOverall Translation Quality (chrF++): {results['chrfpp_score']:.2f}")
    
    # Terminology Success Rate
    if "terminology" in results:
        term_info = results["terminology"]
        
        if "proper_terms" in term_info:
            proper = term_info["proper_terms"]
            report.append(f"\nTerminology Accuracy: {proper['success_rate']:.2%}")
            report.append(f"  Found: {proper['found_terms']}/{proper['total_terms']} terms")
            report.append("\n  Term Details:")
            for source, term_result in proper["term_results"].items():
                status = "✓" if term_result["found"] else "✗"
                report.append(f"    {status} '{source}' -> '{term_result['expected']}'")
    
    report.append("\n" + "=" * 80)
    return "\n".join(report)

