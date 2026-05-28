#!/usr/bin/env python3
"""
Evaluation script for the LLM Council Translation System.

Evaluates translations using:
1. Overall Translation Quality (chrF++)
2. Terminology Accuracy
"""

import json
import argparse
from typing import List, Dict, Optional
from evaluation import (
    evaluate_translation,
    format_evaluation_report
)
from utils import get_terminology_fields


# Test data from user specification
TEST_DATA = [
    {
        "en": "The Governing Council of the Eurosystem decided to continue the APP in February more strongly than its previous decisions.",
        "de": "El Consejo de Gobierno del Eurosistema decidió continuar el APP en febrero en mayor medida que sus decisiones anteriores.",
        "proper_terms": {
            "Governing Council": "Consejo de Gobierno",
            "Eurosystem": "Eurosistema",
            "APP": "APP"
        },
        "random_terms": {
            "February": "febrero",
            "more strongly": "en mayor medida",
            "its": "su"
        }
    },
    {
        "en": "Open the consumption model containing the measures and attributes you want to include in your perspective, and click the Perspectives tab.",
        "de": "Öffnen Sie das Verbrauchsmodell mit den Kennzahlen und Attribute, die Sie in Ihre Perspektive aufnehmen möchten, un wechseln Sie zur Registerkarte Perspektiven.",
        "proper_terms": {
            "consumption model": "Verbrauchsmodell"
        },
        "random_terms": {
            "include": "aufnehmen",
            "want": "möchten"
        }
    }
]


def evaluate_single_translation(
    source: str,
    reference: str,
    hypothesis: str,
    proper_terms: Dict[str, str],
    random_terms: Optional[Dict[str, str]] = None
) -> Dict:
    """Evaluate a single translation."""
    results = evaluate_translation(
        reference=reference,
        hypothesis=hypothesis,
        proper_terms=proper_terms,
        random_terms=random_terms
    )
    results["source"] = source
    results["reference"] = reference
    results["hypothesis"] = hypothesis
    return results


def evaluate_from_council(
    source_text: str,
    council_output: str,
    test_entry: Dict
) -> Dict:
    """
    Evaluate a translation produced by the council system.
    
    Args:
        source_text: Original English text
        council_output: Translation produced by the council
        test_entry: Test data entry with reference and term dictionaries
    """
    proper_terms, random_terms = get_terminology_fields(test_entry)
    return evaluate_single_translation(
        source=source_text,
        reference=test_entry["de"],
        hypothesis=council_output,
        proper_terms=proper_terms,
        random_terms=random_terms
    )


def evaluate_test_set(
    test_data: List[Dict],
    hypotheses: List[str]
) -> Dict:
    """
    Evaluate a set of translations against test data.
    
    Args:
        test_data: List of test entries with references and term dictionaries
        hypotheses: List of system translations (one per test entry)
    
    Returns:
        Aggregated evaluation results
    """
    if len(test_data) != len(hypotheses):
        raise ValueError(f"Test data ({len(test_data)}) and hypotheses ({len(hypotheses)}) length mismatch")
    
    individual_results = []
    all_proper_terms = {}
    
    for test_entry, hypothesis in zip(test_data, hypotheses):
        proper_terms, random_terms = get_terminology_fields(test_entry)
        result = evaluate_single_translation(
            source=test_entry["en"],
            reference=test_entry["de"],
            hypothesis=hypothesis,
            proper_terms=proper_terms,
            random_terms=random_terms
        )
        individual_results.append(result)
        
        # Collect all proper terms for consistency evaluation
        all_proper_terms.update(proper_terms)
    
    # Aggregate metrics
    avg_chrfpp = sum(r["chrfpp_score"] for r in individual_results) / len(individual_results)
    
    # Aggregate terminology success rates
    total_proper_terms = sum(
        r["terminology"]["proper_terms"]["total_terms"]
        for r in individual_results
        if "proper_terms" in r["terminology"]
    )
    found_proper_terms = sum(
        r["terminology"]["proper_terms"]["found_terms"]
        for r in individual_results
        if "proper_terms" in r["terminology"]
    )
    overall_proper_success = found_proper_terms / total_proper_terms if total_proper_terms > 0 else 0.0
    
    return {
        "individual_results": individual_results,
        "aggregated_metrics": {
            "average_chrfpp": avg_chrfpp,
            "overall_terminology_accuracy": overall_proper_success,
            "total_proper_terms": total_proper_terms,
            "found_proper_terms": found_proper_terms
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate translations using chrF++ and Terminology Accuracy"
    )
    parser.add_argument(
        "--reference",
        type=str,
        help="Reference translation (ground truth)"
    )
    parser.add_argument(
        "--hypothesis",
        type=str,
        help="System translation (hypothesis)"
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Source text (English)"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        help="Path to JSON file with test data"
    )
    parser.add_argument(
        "--hypotheses",
        type=str,
        nargs="+",
        help="System translations (for batch evaluation)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    # Single translation evaluation
    if args.reference and args.hypothesis:
        # Load test data if provided, otherwise use defaults
        if args.test_data:
            with open(args.test_data, 'r') as f:
                test_data = json.load(f)
        else:
            test_data = TEST_DATA
        
        # Find matching test entry
        test_entry = None
        if args.source:
            for entry in test_data:
                if entry["en"] == args.source:
                    test_entry = entry
                    break
        
        if not test_entry:
            # Use first entry as default or create minimal entry
            if test_data:
                test_entry = test_data[0]
            else:
                test_entry = {
                    "proper_terms": {},
                    "random_terms": {}
                }
        
        proper_terms, random_terms = get_terminology_fields(test_entry)
        results = evaluate_single_translation(
            source=args.source or "",
            reference=args.reference,
            hypothesis=args.hypothesis,
            proper_terms=proper_terms,
            random_terms=random_terms
        )
        
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(format_evaluation_report(results))
    
    # Batch evaluation
    elif args.test_data and args.hypotheses:
        with open(args.test_data, 'r') as f:
            test_data = json.load(f)
        
        results = evaluate_test_set(test_data, args.hypotheses)
        
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print("=" * 80)
            print("BATCH EVALUATION RESULTS")
            print("=" * 80)
            print(f"\nAverage chrF++: {results['aggregated_metrics']['average_chrfpp']:.2f}")
            print(f"Overall Terminology Accuracy: {results['aggregated_metrics']['overall_terminology_accuracy']:.2%}")
            print(f"Found Terms: {results['aggregated_metrics']['found_proper_terms']}/{results['aggregated_metrics']['total_proper_terms']}")
    
    # Default: evaluate test data
    else:
        print("No evaluation specified. Use --help for usage information.")
        print("\nExample usage:")
        print("  python evaluate.py --reference 'El texto' --hypothesis 'The text' --source 'The text'")
        print("  python evaluate.py --test-data test_data.json --hypotheses 'trans1' 'trans2'")


if __name__ == "__main__":
    main()

