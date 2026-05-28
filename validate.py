#!/usr/bin/env python3
"""
Validation script for the translation pipeline.
Processes a JSONL file and calculates average metrics.
"""

import json
import sys
import os
import argparse
import torch
from typing import List, Dict

# Add the directory containing this file to Python path
_script_dir = os.path.dirname(os.path.realpath(__file__))
if _script_dir and _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline import Pipeline
from evaluation import evaluate_translation
from utils import get_terminology_fields

# Model configuration
MT_MODEL = "facebook/nllb-200-1.3B"
CHECKER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 512


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file (one JSON object per line)."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Validate translation pipeline on JSONL dataset"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/full_data.ende.jsonl",
        help="Path to JSONL validation file (default: data/ende_dev.jsonl)"
    )
    
    # Auto-detect GPU
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            default_device = "auto"
        else:
            default_device = "cuda"
    else:
        default_device = "cpu"
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default=default_device,
        help=f"Device to use (default: {default_device})"
    )
    parser.add_argument(
        "--no-checker",
        action="store_true",
        help="Skip checker stage"
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip first N entries (for resuming)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of entries to process"
    )
    
    args = parser.parse_args()
    
    # Load validation data
    if not os.path.exists(args.data_file):
        print(f"Error: File not found: {args.data_file}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading validation data from: {args.data_file}")
    data = load_jsonl(args.data_file)
    total_entries = len(data)
    
    if args.skip > 0:
        data = data[args.skip:]
        print(f"Skipping first {args.skip} entries")
    
    if args.limit:
        data = data[:args.limit]
        print(f"Limiting to {args.limit} entries")
    
    print(f"Processing {len(data)} entries (out of {total_entries} total)")
    
    # Check if reference translations are available
    has_references = any("de" in entry for entry in data)
    if not has_references:
        print("Note: No reference translations (de) found. Translations will be performed but metrics require references.")
    
    # Initialize pipeline
    print(f"\nInitializing pipeline (Device: {args.device})...")
    pipeline = Pipeline(
        mt_model_id=MT_MODEL,
        checker_model_id=CHECKER_MODEL,
        device=args.device,
        max_new_tokens=MAX_NEW_TOKENS
    )
    
    pipeline.load_models()
    
    # Process all entries
    print(f"\nTranslating {len(data)} entries...")
    results = []
    mt_translations = []
    checker_translations = []
    references = []
    
    for i, entry in enumerate(data, 1):
        source_text = entry.get("en", "").strip()
        if not source_text:
            print(f"Warning: Entry {i} has no 'en' field, skipping")
            continue
        
        reference = entry.get("de", "").strip() if entry.get("de") else None
        proper_terms, _ = get_terminology_fields(entry)
        terms = proper_terms
        
        # Translate
        result = pipeline.run(
            source_text=source_text,
            terms=terms if terms else None,
            skip_checker=args.no_checker
        )
        
        mt_translations.append(result['mt_translation'])
        checker_translations.append(result['final_translation'])
        if reference:
            references.append(reference)
        
        results.append({
            'source': source_text,
            'mt': result['mt_translation'],
            'checker': result['final_translation'],
            'reference': reference,
            'terms': terms
        })
        
        if i % 10 == 0:
            print(f"Processed {i}/{len(data)} entries...", end='\r')
    
    print(f"\nCompleted: {len(results)} entries processed")
    
    # Calculate metrics if references are available
    if has_references and references:
        print("\nCalculating average metrics...")
        
        mt_metrics = []
        checker_metrics = []
        
        for i, result in enumerate(results):
            if result['reference']:
                terms_dict = result.get('terms', {})
                # MT metrics
                mt_eval = evaluate_translation(
                    reference=result['reference'],
                    hypothesis=result['mt'],
                    proper_terms=terms_dict
                )
                mt_metrics.append(mt_eval)
                
                # Checker metrics
                checker_eval = evaluate_translation(
                    reference=result['reference'],
                    hypothesis=result['checker'],
                    proper_terms=terms_dict
                )
                checker_metrics.append(checker_eval)
        
        # Calculate averages
        if mt_metrics:
            avg_mt_chrf = sum(m["chrfpp_score"] for m in mt_metrics) / len(mt_metrics)
            mt_term_acc = []
            for m in mt_metrics:
                if 'terminology' in m and 'proper_terms' in m['terminology']:
                    proper = m['terminology']['proper_terms']
                    if proper['total_terms'] > 0:
                        mt_term_acc.append(proper['success_rate'])
            avg_mt_term = sum(mt_term_acc) / len(mt_term_acc) if mt_term_acc else None
            
            print(f"\nMT Model Average Metrics (n={len(mt_metrics)}):")
            print(f"  chrF++: {avg_mt_chrf:.2f}")
            if avg_mt_term is not None:
                print(f"  Terminology Accuracy: {avg_mt_term:.1%}")
        
        if checker_metrics and not args.no_checker:
            avg_checker_chrf = sum(c["chrfpp_score"] for c in checker_metrics) / len(checker_metrics)
            checker_term_acc = []
            for c in checker_metrics:
                if 'terminology' in c and 'proper_terms' in c['terminology']:
                    proper = c['terminology']['proper_terms']
                    if proper['total_terms'] > 0:
                        checker_term_acc.append(proper['success_rate'])
            avg_checker_term = sum(checker_term_acc) / len(checker_term_acc) if checker_term_acc else None
            
            print(f"\nChecker Model Average Metrics (n={len(checker_metrics)}):")
            print(f"  chrF++: {avg_checker_chrf:.2f}")
            if avg_checker_term is not None:
                print(f"  Terminology Accuracy: {avg_checker_term:.1%}")
            
            if mt_metrics:
                chrf_improvement = avg_checker_chrf - avg_mt_chrf
                print(f"\nImprovement:")
                print(f"  chrF++: {chrf_improvement:+.2f}")
                if avg_mt_term is not None and avg_checker_term is not None:
                    term_improvement = avg_checker_term - avg_mt_term
                    print(f"  Terminology Accuracy: {term_improvement:+.1%}")
    else:
        print("\nNo reference translations available. Metrics cannot be calculated.")
        print(f"Translated {len(results)} entries successfully.")
    
    print("\nValidation complete.")


if __name__ == "__main__":
    main()
