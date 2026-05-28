"""Evaluation metrics: chrF++ and terminology success rate."""

from typing import Dict, List, Tuple, Optional
from collections import Counter
import re


def calculate_chrfpp(reference: str, hypothesis: str) -> float:
    """chrF++ (word_order=2) via sacrebleu; score is in 0–100 range."""
    try:
        import sacrebleu
        chrf = sacrebleu.sentence_chrf(hypothesis, [reference], word_order=2)
        return chrf.score
    except ImportError:
        raise ImportError("sacrebleu is required. Install with: pip install sacrebleu")


def terminology_success_rate(
    hypothesis: str,
    proper_terms: Dict[str, str],
    case_sensitive: bool = False
) -> Tuple[float, Dict[str, bool]]:
    """Fraction of expected German terms that appear in the hypothesis (word-boundary match)."""
    if not proper_terms:
        return 1.0, {}

    term_results = {}
    found_count = 0
    text = hypothesis if case_sensitive else hypothesis.lower()

    for source_term, expected_translation in proper_terms.items():
        expected = expected_translation if case_sensitive else expected_translation.lower()
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
    Percentage of translations that use the most common rendering of a source term.

    Without expected_translation this cannot determine consistency (proper alignment
    tools would be needed), so all entries are marked None.
    """
    if not hypotheses:
        return 0.0, {}

    translations = []
    for hyp in hypotheses:
        if expected_translation:
            pattern = r'\b' + re.escape(expected_translation) + r'\b'
            if re.search(pattern, hyp, re.IGNORECASE):
                translations.append(expected_translation)
            else:
                translations.append(None)
        else:
            translations.append(None)

    translation_counts = Counter(translations)
    if not translation_counts:
        return 0.0, {}

    _, most_common_count = translation_counts.most_common(1)[0]
    consistency_score = most_common_count / len(translations)
    return consistency_score, dict(translation_counts)


def evaluate_translation(
    reference: str,
    hypothesis: str,
    proper_terms: Dict[str, str],
    random_terms: Optional[Dict[str, str]] = None
) -> Dict:
    """chrF++ + terminology success rate for a single translation."""
    results = {
        "chrfpp_score": calculate_chrfpp(reference, hypothesis),
        "terminology": {}
    }
    proper_success_rate, proper_term_results = terminology_success_rate(hypothesis, proper_terms)
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
    """Consistency metrics per source term across a list of translations."""
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
    """Format evaluation results as a readable report string."""
    report = []
    report.append("=" * 80)
    report.append("EVALUATION REPORT")
    report.append("=" * 80)
    report.append(f"\nOverall Translation Quality (chrF++): {results['chrfpp_score']:.2f}")

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
