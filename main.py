#!/usr/bin/env python3
"""
MT + LLM-checker translation pipeline.

Usage:
  python main.py --data-file data/ende_dev.jsonl --max-items 100
  python main.py --data-file data/ende_dev.jsonl --disable-checker
  python main.py --data-file data/ende_dev.jsonl --mt-model facebook/nllb-moe-54b --diagnose --num-examples 5 --disable-checker
  python main.py --text "Hello world" --mt-model facebook/nllb-200-3.3B
  python main.py --data-file data/ende_dev.jsonl --offline --seed 42
"""

import argparse
import sys
import json
import os
import time
import socket
import subprocess
import torch
import random
import numpy as np

_script_dir = os.path.dirname(os.path.realpath(__file__))
if _script_dir and _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline import Pipeline
from utils import load_terms_from_json, get_terminology_fields
from evaluation import evaluate_translation


def load_jsonl(file_path: str) -> list:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_git_commit():
    """Get current git commit hash if available."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=_script_dir,
            capture_output=True,
            text=True,
            timeout=1
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def log_system_info(args):
    """Log system and configuration info."""
    import transformers
    
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    hostname = socket.gethostname()
    git_commit = get_git_commit()
    
    print(f"Run: {timestamp} | {hostname}", end="")
    if git_commit:
        print(f" | git:{git_commit}", end="")
    print()
    
    max_items_str = str(args.max_items) if args.max_items else "all"
    print(f"Args: data={args.data_file}, max_items={max_items_str}, num_examples={args.num_examples}, batch={args.batch_size}")
    print(f"      mt={args.mt_model}, checker={args.checker_model if not args.disable_checker else 'disabled'}")
    print(f"      device={args.device}, dtype={args.dtype}, seed={args.seed}, terminology_mode={args.terminology_mode}")
    
    # Device info
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    print(f"Effective devices: CUDA_VISIBLE_DEVICES={visible_devices}")
    print(f"Versions: torch={torch.__version__}, transformers={transformers.__version__}")
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"GPUs: {num_gpus}", end="")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / 1e9
            print(f", GPU{i}={props.name} ({vram_gb:.1f}GB)", end="")
        print()
    else:
        print("GPUs: none (CPU mode)")
    
    print(f"Model: {args.mt_model}, dtype={args.dtype}, max_new_tokens={args.max_new_tokens}")
    print(f"Config: compile={args.compile}")


def set_deterministic(seed: int):
    """Set deterministic random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def truncate_text(text: str, max_len: int = 500) -> str:
    """Truncate text for display."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...[truncated]"


def run_diagnostics(results: list, num_examples: int = 10):
    """Print diagnostic examples."""
    results_with_refs = [r for r in results if r.get('reference')]
    if not results_with_refs:
        print("\nNo results with references for diagnostics.")
        return
    
    if len(results_with_refs) > num_examples:
        sample = random.sample(results_with_refs, num_examples)
    else:
        sample = results_with_refs
        num_examples = len(sample)
    
    print(f"\nDiagnostics: {num_examples} examples")
    print("-" * 80)
    
    for i, result in enumerate(sample, 1):
        source = result['source']
        mt_output = result['mt']
        checker_output = result['checker']
        reference = result['reference']
        terms = result.get('terms', {})
        
        mt_eval = evaluate_translation(reference, mt_output, terms)
        checker_eval = evaluate_translation(reference, checker_output, terms)
        
        print(f"\nExample {i}/{num_examples}:")
        print(f"  Source: {truncate_text(source)}")
        print(f"  Reference: {truncate_text(reference)}")
        print(f"  MT: {truncate_text(mt_output)} (chrF++: {mt_eval['chrfpp_score']:.2f})")
        print(f"  Checker: {truncate_text(checker_output)} (chrF++: {checker_eval['chrfpp_score']:.2f})")
        
        if terms:
            mt_term = mt_eval.get('terminology', {}).get('proper_terms', {}).get('success_rate', 0.0)
            checker_term = checker_eval.get('terminology', {}).get('proper_terms', {}).get('success_rate', 0.0)
            print(f"  Terms: MT={mt_term:.1%}, Checker={checker_term:.1%}")


def run_validation(args):
    """Run validation on dataset."""
    stage_start = time.time()
    
    # Stage A: Load data
    print(f"\n[Stage A] Loading data...")
    if not os.path.exists(args.data_file):
        print(f"Error: File not found: {args.data_file}", file=sys.stderr)
        sys.exit(1)
    
    data = load_jsonl(args.data_file)
    total_entries = len(data)
    
    if args.skip > 0:
        data = data[args.skip:]
        print(f"  Skipped {args.skip} entries")
    
    if args.max_items:
        data = data[:args.max_items]
        print(f"  Limited to {args.max_items} entries")
    
    processed_items = len(data)
    print(f"  Dataset: total={total_entries}, processed={processed_items} in {time.time() - stage_start:.1f}s")
    
    has_references = any("de" in entry for entry in data)
    if not has_references:
        print("  Note: No reference translations found. Metrics require references.")
    
    # Prepare entries
    entries_to_process = []
    for i, entry in enumerate(data, 1):
        source_text = entry.get("en", "").strip()
        if not source_text:
            continue
        proper_terms, _ = get_terminology_fields(entry)
        entries_to_process.append({
            'index': i,
            'source': source_text,
            'reference': entry.get("de", "").strip() if entry.get("de") else None,
            'terms': proper_terms
        })
    
    # Stage B: Load models
    print(f"\n[Stage B] Loading models...")
    stage_start = time.time()
    
    pipeline = Pipeline(
        mt_model_id=args.mt_model,
        checker_model_id=args.checker_model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        disable_checker=args.disable_checker,
        cache_dir=args.cache_dir,
        compile_model=args.compile,
        terminology_mode=args.terminology_mode
    )
    
    pipeline.load_models()
    print(f"  Models loaded in {time.time() - stage_start:.1f}s")
    
    # Stage C: Run inference
    print(f"\n[Stage C] Running inference...")
    stage_start = time.time()
    results = []
    total_batches = (len(entries_to_process) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(0, len(entries_to_process), args.batch_size):
        batch_entries = entries_to_process[batch_idx:batch_idx + args.batch_size]
        source_texts = [e['source'] for e in batch_entries]
        terms_list = [e['terms'] if e['terms'] else None for e in batch_entries]
        
        batch_start = time.time()
        batch_results = pipeline.run_batch(
            source_texts=source_texts,
            terms_list=terms_list,
            skip_checker=args.disable_checker
        )
        batch_time = time.time() - batch_start
        
        for entry, result in zip(batch_entries, batch_results):
            results.append({
                'source': entry['source'],
                'mt': result['mt_translation'],
                'checker': result['final_translation'],
                'reference': entry['reference'],
                'terms': entry['terms'],
                'checker_raw': result.get('checker_result', {}).get('raw_output', None) if result.get('checker_result') else None
            })
        
        batch_num = (batch_idx // args.batch_size) + 1
        items_per_sec = len(batch_entries) / batch_time if batch_time > 0 else 0
        if batch_num % 10 == 0 or batch_num == total_batches:
            elapsed = time.time() - stage_start
            print(f"  Batch {batch_num}/{total_batches}: {items_per_sec:.1f} items/s, {batch_time:.2f}s/batch, {elapsed:.1f}s elapsed")
    
    print(f"  Inference complete: {len(results)} items in {time.time() - stage_start:.1f}s")
    
    # Stage D: Compute metrics
    print(f"\n[Stage D] Computing metrics...")
    stage_start = time.time()
    
    # DIAGNOSTIC: Check if checker outputs actually differ from MT
    diff_count = sum(1 for r in results if r.get('mt', '').strip() != r.get('checker', '').strip())
    print(f"  [DIAG] MT vs Checker differences: {diff_count}/{len(results)}")
    
    if diff_count > 0:
        # Show 2 examples where they differ
        diff_examples = [(i, r) for i, r in enumerate(results) if r.get('mt', '').strip() != r.get('checker', '').strip()]
        for idx, (i, r) in enumerate(diff_examples[:2], 1):
            print(f"  [DIAG] Example {idx} (index {i}):")
            print(f"    Source: {truncate_text(r.get('source', ''), 80)}")
            print(f"    MT:     {truncate_text(r.get('mt', ''), 80)}")
            print(f"    Checker: {truncate_text(r.get('checker', ''), 80)}")
        
        # Hash check for metrics inputs
        import hashlib
        mt_hyps = [r.get('mt', '') for r in results if r.get('reference')]
        checker_hyps = [r.get('checker', '') for r in results if r.get('reference')]
        mt_hash = hashlib.md5('|'.join(mt_hyps).encode()).hexdigest()[:16]
        checker_hash = hashlib.md5('|'.join(checker_hyps).encode()).hexdigest()[:16]
        print(f"  [DIAG] MT hypotheses hash: {mt_hash}")
        print(f"  [DIAG] Checker hypotheses hash: {checker_hash}")
        if mt_hash == checker_hash:
            print(f"  [DIAG] WARNING: Hashes are identical! Metrics will be identical.")
        else:
            print(f"  [DIAG] Hashes differ - metrics should differ if computed correctly.")
    
    if has_references and any(r['reference'] for r in results):
        mt_metrics = []
        checker_metrics = []
        
        for result in results:
            if result['reference']:
                terms_dict = result.get('terms', {})
                mt_eval = evaluate_translation(result['reference'], result['mt'], terms_dict)
                mt_metrics.append(mt_eval)
                
                checker_eval = evaluate_translation(result['reference'], result['checker'], terms_dict)
                checker_metrics.append(checker_eval)
        
        if mt_metrics:
            avg_mt_chrf = sum(m["chrfpp_score"] for m in mt_metrics) / len(mt_metrics)
            mt_term_acc = []
            for m in mt_metrics:
                if 'terminology' in m and 'proper_terms' in m['terminology']:
                    proper = m['terminology']['proper_terms']
                    if proper['total_terms'] > 0:
                        mt_term_acc.append(proper['success_rate'])
            avg_mt_term = sum(mt_term_acc) / len(mt_term_acc) if mt_term_acc else None
            
            print(f"  MT: chrF++={avg_mt_chrf:.2f}", end="")
            if avg_mt_term is not None:
                print(f", term_acc={avg_mt_term:.1%}")
            else:
                print()
        
        if args.disable_checker:
            print(f"  Checker: disabled (using MT output)")
            print(f"  Improvement: chrF++=+0.00", end="")
            if avg_mt_term is not None:
                print(f", term_acc=+0.0%")
            else:
                print()
        elif checker_metrics:
            avg_checker_chrf = sum(c["chrfpp_score"] for c in checker_metrics) / len(checker_metrics)
            checker_term_acc = []
            for c in checker_metrics:
                if 'terminology' in c and 'proper_terms' in c['terminology']:
                    proper = c['terminology']['proper_terms']
                    if proper['total_terms'] > 0:
                        checker_term_acc.append(proper['success_rate'])
            avg_checker_term = sum(checker_term_acc) / len(checker_term_acc) if checker_term_acc else None
            
            print(f"  Checker: chrF++={avg_checker_chrf:.2f}", end="")
            if avg_checker_term is not None:
                print(f", term_acc={avg_checker_term:.1%}")
            else:
                print()
            
            if mt_metrics:
                chrf_improvement = avg_checker_chrf - avg_mt_chrf
                print(f"  Improvement: chrF++={chrf_improvement:+.2f}", end="")
                if avg_mt_term is not None and avg_checker_term is not None:
                    term_improvement = avg_checker_term - avg_mt_term
                    print(f", term_acc={term_improvement:+.1%}")
                else:
                    print()
    else:
        print("  No reference translations available. Metrics skipped.")
    
    print(f"  Metrics computed in {time.time() - stage_start:.1f}s")
    
    if args.diagnose:
        num_diagnostic_samples = args.num_examples
        print(f"  Diagnostics: processed={len(results)}, sample_count={num_diagnostic_samples}")
        run_diagnostics(results, num_diagnostic_samples)
    
    print("\nComplete.")


def translate_single_sentence(args):
    """Translate a single sentence."""
    source_text = args.text
    
    pipeline = Pipeline(
        mt_model_id=args.mt_model,
        checker_model_id=args.checker_model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        disable_checker=args.disable_checker,
        cache_dir=args.cache_dir,
        compile_model=args.compile,
        terminology_mode=args.terminology_mode
    )
    
    pipeline.load_models()
    
    terms = None
    if args.terms:
        terms = load_terms_from_json(args.terms)
    
    result = pipeline.run(
        source_text=source_text,
        terms=terms,
        skip_checker=args.disable_checker
    )
    
    print(f"\nSource: {result['source_text']}")
    print(f"MT: {result['mt_translation']}")
    if args.disable_checker:
        print(f"Final: {result['final_translation']}")
    else:
        print(f"Checker: {result['final_translation']}")


def main():
    parser = argparse.ArgumentParser(description="MT + LLM-checker translation pipeline")
    
    # Data arguments
    parser.add_argument("--text", type=str, default=None, help="Single sentence to translate")
    parser.add_argument("--data-file", type=str, default="data/full_data.ende.jsonl", help="Path to JSONL data file")
    parser.add_argument("--max-items", type=int, default=None, help="Limit number of dataset rows to process (default: all)")
    parser.add_argument("--num-examples", type=int, default=20, help="Number of diagnostic examples to print (default: 20)")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N entries")
    
    # Model arguments
    parser.add_argument("--mt-model", type=str, default="facebook/nllb-200-3.3B", help="MT model identifier")
    parser.add_argument("--checker-model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Checker model identifier")
    parser.add_argument("--disable-checker", action="store_true", help="Disable checker stage")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens for checker generation")
    
    # Processing arguments
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing")
    
    # Device arguments
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            default_device = "auto"
        else:
            default_device = "cuda"
    else:
        default_device = "cpu"
    
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], default=default_device, help="Device to use")
    parser.add_argument("--dtype", type=str, choices=["fp32", "fp16", "bf16", "auto"], default="auto", help="Data type")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (experimental)")
    
    # HPC arguments
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--cache-dir", type=str, default=None, help="Hugging Face cache directory")
    parser.add_argument("--offline", action="store_true", help="Offline mode (no downloads)")
    
    # Other arguments
    parser.add_argument("--terms", type=str, default=None, help="Path to terminology JSON file")
    parser.add_argument("--diagnose", action="store_true", help="Show diagnostic examples")
    parser.add_argument("--no-checker", action="store_true", help="[Deprecated] Use --disable-checker")
    parser.add_argument("--terminology-mode", type=str, choices=["on", "off"], default="on", help="Terminology mode: 'on' to enforce terminology, 'off' to ignore it (default: on)")
    
    args = parser.parse_args()
    
    # Handle deprecated flag
    if args.no_checker:
        args.disable_checker = True
    
    # Check environment variable (CLI takes precedence)
    env_disable_checker = os.environ.get("MTLLM_DISABLE_CHECKER", "0") in ("1", "true", "True", "TRUE")
    if not args.disable_checker:
        args.disable_checker = env_disable_checker
    
    # Set offline mode
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    # Set cache directory
    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
    
    # Set deterministic seed
    if args.seed is not None:
        set_deterministic(args.seed)
    
    # Log system info
    log_system_info(args)
    
    # Run pipeline
    try:
        if args.text:
            translate_single_sentence(args)
        else:
            run_validation(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n\nError: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
