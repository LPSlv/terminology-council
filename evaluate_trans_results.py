#!/usr/bin/env python3
"""
Comprehensive evaluation script for translation results in data/trans folder.

Evaluates all translation result files against reference data and generates
comparative reports.
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import sys

# Add the directory containing this file to Python path
_script_dir = os.path.dirname(os.path.realpath(__file__))
if _script_dir and _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from utils import get_terminology_fields

# Check for sacrebleu dependency (but allow --help to work)
def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import sacrebleu
    except ImportError:
        print("ERROR: sacrebleu is required for evaluation.", file=sys.stderr)
        print("Install it with: pip install sacrebleu", file=sys.stderr)
        print("Or install all requirements: pip install -r requirements.txt", file=sys.stderr)
        return False
    try:
        from evaluate import evaluate_single_translation
    except ImportError as e:
        print(f"Error importing evaluate module: {e}", file=sys.stderr)
        return False
    return True


def load_jsonl(filepath: str) -> List[Dict]:
    """Load a JSONL file and return list of dictionaries."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_reference_data(reference_path: str) -> List[Dict]:
    """Load reference data with translations and terminology."""
    return load_jsonl(reference_path)


def evaluate_translation_file(
    translation_file: str,
    reference_data: List[Dict],
    match_by_source: bool = True,
    evaluate_fn=None
) -> Dict:
    """
    Evaluate a translation file against reference data.
    
    Args:
        translation_file: Path to translation result file
        reference_data: List of reference entries with 'en', 'de', 'proper_terms'
        match_by_source: If True, match translations by source text; otherwise by index
    
    Returns:
        Dictionary with evaluation results
    """
    translations = load_jsonl(translation_file)
    
    if len(translations) != len(reference_data):
        print(f"Warning: Translation file has {len(translations)} entries, "
              f"reference has {len(reference_data)} entries", file=sys.stderr)
    
    individual_results = []
    total_proper_terms = 0
    found_proper_terms = 0
    chrfpp_scores = []
    
    # Create a lookup for reference data if matching by source
    if match_by_source:
        ref_lookup = {entry['en']: entry for entry in reference_data}
    else:
        ref_lookup = None
    
    for idx, trans_entry in enumerate(translations):
        source_text = trans_entry.get('en', '')
        hypothesis = trans_entry.get('de', '')
        
        # Find matching reference entry
        if match_by_source and ref_lookup:
            ref_entry = ref_lookup.get(source_text)
            if not ref_entry:
                # Try to find by index as fallback
                if idx < len(reference_data):
                    ref_entry = reference_data[idx]
                else:
                    continue
        else:
            if idx < len(reference_data):
                ref_entry = reference_data[idx]
            else:
                continue
        
        reference = ref_entry.get('de', '')
        proper_terms, random_terms = get_terminology_fields(ref_entry)
        
        if not reference or not hypothesis:
            continue
        
        # Evaluate this translation using evaluate_single_translation from evaluate.py
        try:
            if evaluate_fn is None:
                from evaluate import evaluate_single_translation
                evaluate_fn = evaluate_single_translation
            result = evaluate_fn(
                source=source_text,
                reference=reference,
                hypothesis=hypothesis,
                proper_terms=proper_terms,
                random_terms=random_terms
            )
            
            individual_results.append({
                'index': idx,
                'source': source_text[:100] + '...' if len(source_text) > 100 else source_text,
                'chrfpp': result['chrfpp_score'],
                'terminology_success': result['terminology']['proper_terms']['success_rate'],
                'found_terms': result['terminology']['proper_terms']['found_terms'],
                'total_terms': result['terminology']['proper_terms']['total_terms']
            })
            
            chrfpp_scores.append(result['chrfpp_score'])
            total_proper_terms += result['terminology']['proper_terms']['total_terms']
            found_proper_terms += result['terminology']['proper_terms']['found_terms']
            
        except Exception as e:
            print(f"Error evaluating entry {idx} in {translation_file}: {e}", file=sys.stderr)
            continue
    
    # Calculate aggregated metrics
    avg_chrfpp = sum(chrfpp_scores) / len(chrfpp_scores) if chrfpp_scores else 0.0
    overall_term_accuracy = found_proper_terms / total_proper_terms if total_proper_terms > 0 else 0.0
    
    return {
        'file': translation_file,
        'num_evaluated': len(individual_results),
        'aggregated_metrics': {
            'average_chrfpp': avg_chrfpp,
            'overall_terminology_accuracy': overall_term_accuracy,
            'total_proper_terms': total_proper_terms,
            'found_proper_terms': found_proper_terms
        },
        'individual_results': individual_results
    }


def find_translation_files(base_dir: str) -> Dict[str, List[str]]:
    """
    Find all translation result files organized by folder.
    
    Returns:
        Dictionary mapping folder name to list of file paths
    """
    base_path = Path(base_dir)
    result_files = defaultdict(list)
    
    # Find all result folders
    for folder in base_path.iterdir():
        if folder.is_dir():
            folder_name = folder.name
            # Find all .jsonl files in this folder
            for file in folder.glob('*.jsonl'):
                # Skip review files (rev_by_*.jsonl) for now
                if 'rev_by' not in file.name:
                    result_files[folder_name].append(str(file))
    
    return dict(result_files)


def compare_results(all_results: Dict[str, Dict]) -> Dict:
    """
    Compare results across different files and folders.
    
    Args:
        all_results: Dictionary mapping file path to evaluation results
    
    Returns:
        Comparison summary
    """
    comparison = {
        'by_folder': defaultdict(list),
        'by_system': defaultdict(list),
        'with_reviewer': [],
        'without_reviewer': []
    }
    
    for filepath, results in all_results.items():
        path = Path(filepath)
        folder_name = path.parent.name
        filename = path.name
        
        # Extract system name (e.g., 'cand_a', 'my_system')
        if 'cand_a' in filename:
            system = 'cand_a'
        elif 'cand_b' in filename:
            system = 'cand_b'
        elif 'cand_c' in filename:
            system = 'cand_c'
        elif 'my_system' in filename:
            system = 'my_system'
        else:
            system = 'unknown'
        
        metrics = results['aggregated_metrics']
        
        comparison['by_folder'][folder_name].append({
            'file': filename,
            'system': system,
            'chrfpp': metrics['average_chrfpp'],
            'term_accuracy': metrics['overall_terminology_accuracy']
        })
        
        comparison['by_system'][system].append({
            'folder': folder_name,
            'file': filename,
            'chrfpp': metrics['average_chrfpp'],
            'term_accuracy': metrics['overall_terminology_accuracy']
        })
        
        # Categorize by reviewer presence
        if 'norev' in folder_name:
            comparison['without_reviewer'].append({
                'file': filepath,
                'system': system,
                'chrfpp': metrics['average_chrfpp'],
                'term_accuracy': metrics['overall_terminology_accuracy']
            })
        else:
            comparison['with_reviewer'].append({
                'file': filepath,
                'system': system,
                'chrfpp': metrics['average_chrfpp'],
                'term_accuracy': metrics['overall_terminology_accuracy']
            })
    
    return comparison


def format_comparison_report(comparison: Dict, all_results: Dict[str, Dict]) -> str:
    """Format a comprehensive comparison report."""
    report = []
    report.append("=" * 100)
    report.append("TRANSLATION RESULTS EVALUATION REPORT")
    report.append("=" * 100)
    
    # Summary by folder
    report.append("\n" + "=" * 100)
    report.append("RESULTS BY FOLDER")
    report.append("=" * 100)
    
    for folder_name, files in sorted(comparison['by_folder'].items()):
        report.append(f"\n📁 {folder_name}")
        report.append("-" * 100)
        for file_info in sorted(files, key=lambda x: x['chrfpp'], reverse=True):
            report.append(
                f"  {file_info['system']:12s} | "
                f"chrF++: {file_info['chrfpp']:6.2f} | "
                f"Term Accuracy: {file_info['term_accuracy']:6.2%}"
            )
    
    # Summary by system
    report.append("\n" + "=" * 100)
    report.append("RESULTS BY SYSTEM")
    report.append("=" * 100)
    
    for system, files in sorted(comparison['by_system'].items()):
        report.append(f"\n🔧 {system}")
        report.append("-" * 100)
        for file_info in sorted(files, key=lambda x: x['chrfpp'], reverse=True):
            report.append(
                f"  {file_info['folder']:25s} | "
                f"chrF++: {file_info['chrfpp']:6.2f} | "
                f"Term Accuracy: {file_info['term_accuracy']:6.2%}"
            )
    
    # With vs Without Reviewer comparison
    report.append("\n" + "=" * 100)
    report.append("WITH REVIEWER vs WITHOUT REVIEWER")
    report.append("=" * 100)
    
    if comparison['with_reviewer']:
        report.append("\n✅ WITH REVIEWER:")
        avg_chrfpp_rev = sum(f['chrfpp'] for f in comparison['with_reviewer']) / len(comparison['with_reviewer'])
        avg_term_rev = sum(f['term_accuracy'] for f in comparison['with_reviewer']) / len(comparison['with_reviewer'])
        report.append(f"  Average chrF++: {avg_chrfpp_rev:.2f}")
        report.append(f"  Average Term Accuracy: {avg_term_rev:.2%}")
    
    if comparison['without_reviewer']:
        report.append("\n❌ WITHOUT REVIEWER:")
        avg_chrfpp_norev = sum(f['chrfpp'] for f in comparison['without_reviewer']) / len(comparison['without_reviewer'])
        avg_term_norev = sum(f['term_accuracy'] for f in comparison['without_reviewer']) / len(comparison['without_reviewer'])
        report.append(f"  Average chrF++: {avg_chrfpp_norev:.2f}")
        report.append(f"  Average Term Accuracy: {avg_term_norev:.2%}")
    
    # Detailed results for each file
    report.append("\n" + "=" * 100)
    report.append("DETAILED RESULTS")
    report.append("=" * 100)
    
    for filepath, results in sorted(all_results.items()):
        report.append(f"\n📄 {filepath}")
        report.append("-" * 100)
        metrics = results['aggregated_metrics']
        report.append(f"  Evaluated entries: {results['num_evaluated']}")
        report.append(f"  Average chrF++: {metrics['average_chrfpp']:.2f}")
        report.append(f"  Overall Terminology Accuracy: {metrics['overall_terminology_accuracy']:.2%}")
        report.append(f"  Found Terms: {metrics['found_proper_terms']}/{metrics['total_proper_terms']}")
    
    report.append("\n" + "=" * 100)
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate translation results from data/trans folder"
    )
    parser.add_argument(
        '--reference',
        type=str,
        default='data/full_data.ende.jsonl',
        help='Path to reference data file (default: data/ende_dev.jsonl)'
    )
    parser.add_argument(
        '--trans-dir',
        type=str,
        default='data/trans',
        help='Directory containing translation results (default: data/trans)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for JSON results (optional)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON instead of formatted report'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Evaluate a specific file instead of all files'
    )
    
    args = parser.parse_args()
    
    # Check dependencies (skip if just showing help)
    if not check_dependencies():
        sys.exit(1)
    
    # Load reference data
    print(f"Loading reference data from {args.reference}...", file=sys.stderr)
    reference_data = load_reference_data(args.reference)
    print(f"Loaded {len(reference_data)} reference entries", file=sys.stderr)
    
    # Find translation files
    if args.file:
        translation_files = [args.file]
    else:
        print(f"Finding translation files in {args.trans_dir}...", file=sys.stderr)
        files_by_folder = find_translation_files(args.trans_dir)
        translation_files = []
        for folder, files in files_by_folder.items():
            translation_files.extend(files)
        print(f"Found {len(translation_files)} translation files", file=sys.stderr)
    
    # Import evaluation function once
    from evaluate import evaluate_single_translation
    
    # Evaluate each file
    all_results = {}
    for trans_file in sorted(translation_files):
        print(f"Evaluating {trans_file}...", file=sys.stderr)
        try:
            results = evaluate_translation_file(trans_file, reference_data, evaluate_fn=evaluate_single_translation)
            all_results[trans_file] = results
        except Exception as e:
            print(f"Error evaluating {trans_file}: {e}", file=sys.stderr)
            continue
    
    # Compare results
    comparison = compare_results(all_results)
    
    # Output results
    if args.json:
        output = {
            'all_results': all_results,
            'comparison': comparison
        }
        print(json.dumps(output, indent=2))
    else:
        report = format_comparison_report(comparison, all_results)
        print(report)
    
    # Save to file if requested
    if args.output:
        output_data = {
            'all_results': all_results,
            'comparison': comparison
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

