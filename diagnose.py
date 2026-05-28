#!/usr/bin/env python3
"""
Diagnostic script to inspect checker outputs and identify evaluation issues.

Prints 10 random examples showing:
- source
- MT output
- checker raw output
- checker extracted final_translation
- reference
- chrF++ per-sentence
"""

import argparse
import json
import random
import sys
import os

# Add the directory containing this file to Python path
_script_dir = os.path.dirname(os.path.realpath(__file__))
if _script_dir and _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline import Pipeline
from evaluation import evaluate_translation
from utils import get_terminology_fields

# Model configuration (same as main.py)
MT_MODEL = "facebook/nllb-200-1.3B"
CHECKER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 512


def load_jsonl(file_path: str) -> list:
    """Load JSONL file (one JSON object per line)."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def diagnose_checker_outputs(
    data_file: str,
    num_examples: int = 10,
    device: str = "cpu",
    limit: int = None,
    skip: int = 0
):
    """Run diagnostic on checker outputs."""
    
    # Load data
    print(f"Loading data from: {data_file}")
    data = load_jsonl(data_file)
    total_entries = len(data)
    
    if skip > 0:
        data = data[skip:]
        print(f"Skipping first {skip} entries")
    
    if limit:
        data = data[:limit]
        print(f"Limiting to {limit} entries")
    
    # Filter to entries with references
    data_with_refs = [entry for entry in data if entry.get("de")]
    
    if not data_with_refs:
        print("Error: No entries with reference translations found")
        return
    
    # Sample random examples
    if len(data_with_refs) > num_examples:
        sample = random.sample(data_with_refs, num_examples)
    else:
        sample = data_with_refs
        num_examples = len(sample)
    
    print(f"\nSampling {num_examples} examples from {len(data_with_refs)} entries with references")
    print(f"Initializing pipeline (Device: {device})...")
    
    # Initialize pipeline
    pipeline = Pipeline(
        mt_model_id=MT_MODEL,
        checker_model_id=CHECKER_MODEL,
        device=device,
        max_new_tokens=MAX_NEW_TOKENS
    )
    
    pipeline.load_models()
    
    print(f"\nProcessing {num_examples} examples...\n")
    print("=" * 100)
    
    for i, entry in enumerate(sample, 1):
        source_text = entry.get("en", "").strip()
        reference = entry.get("de", "").strip()
        proper_terms, _ = get_terminology_fields(entry)
        terms = proper_terms
        
        if not source_text or not reference:
            continue
        
        # Run pipeline
        result = pipeline.run(
            source_text=source_text,
            terms=terms if terms else None,
            skip_checker=False
        )
        
        mt_output = result['mt_translation']
        checker_result = result.get('checker_result', {})
        checker_raw = checker_result.get('raw_output', 'N/A')
        checker_extracted = result['final_translation']
        
        # Calculate metrics
        mt_eval = evaluate_translation(
            reference=reference,
            hypothesis=mt_output,
            proper_terms=terms
        )
        
        checker_eval = evaluate_translation(
            reference=reference,
            hypothesis=checker_extracted,
            proper_terms=terms
        )
        
        # Print diagnostic info
        print(f"\n{'='*100}")
        print(f"EXAMPLE {i}/{num_examples}")
        print(f"{'='*100}\n")
        
        print(f"SOURCE:")
        print(f"  {source_text}\n")
        
        print(f"REFERENCE:")
        print(f"  {reference}\n")
        
        print(f"MT OUTPUT:")
        print(f"  {mt_output}")
        print(f"  chrF++: {mt_eval['chrfpp_score']:.2f}\n")
        
        print(f"CHECKER RAW OUTPUT:")
        print(f"  {repr(checker_raw[:500])}")  # First 500 chars, with repr to show special chars
        if len(checker_raw) > 500:
            print(f"  ... (truncated, total length: {len(checker_raw)})\n")
        else:
            print()
        
        print(f"CHECKER EXTRACTED (final_translation):")
        print(f"  {checker_extracted}")
        print(f"  chrF++: {checker_eval['chrfpp_score']:.2f}\n")
        
        # Check if raw output looks like JSON or contains prompt text
        raw_looks_like_json = checker_raw.strip().startswith('{') or '"final_translation"' in checker_raw
        raw_has_prompt = any(word in checker_raw.lower() for word in ['review', 'translation', 'improve', 'corrected'])
        
        print(f"DIAGNOSTIC FLAGS:")
        print(f"  Raw output looks like JSON: {raw_looks_like_json}")
        print(f"  Raw output contains prompt text: {raw_has_prompt}")
        print(f"  Extracted length: {len(checker_extracted)} chars")
        print(f"  Raw length: {len(checker_raw)} chars")
        print(f"  Extraction changed text: {checker_raw != checker_extracted}")
        
        # Check if extracted is same as MT (fallback used)
        if checker_extracted == mt_output:
            print(f"  ⚠️  WARNING: Extracted translation equals MT output (fallback may have been used)")
        
        # Check terminology
        if terms:
            mt_term_acc = mt_eval['terminology']['proper_terms']['success_rate'] if 'terminology' in mt_eval else 0.0
            checker_term_acc = checker_eval['terminology']['proper_terms']['success_rate'] if 'terminology' in checker_eval else 0.0
            print(f"\nTERMINOLOGY:")
            print(f"  MT accuracy: {mt_term_acc:.1%}")
            print(f"  Checker accuracy: {checker_term_acc:.1%}")
            print(f"  Terms: {terms}")
        
        print()
    
    print("=" * 100)
    print("\nDiagnostic complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose checker outputs to identify evaluation issues"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/full_data.ende.jsonl",
        help="Path to JSONL validation file (default: data/ende_dev.jsonl)"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of random examples to show (default: 10)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Device to use (default: cpu)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of entries to process"
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip first N entries"
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    diagnose_checker_outputs(
        data_file=args.data_file,
        num_examples=args.num_examples,
        device=args.device,
        limit=args.limit,
        skip=args.skip
    )


if __name__ == "__main__":
    main()
