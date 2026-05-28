"""Command-line interface for mt_llm.

Subcommands:
    translate   2-stage MT + LLM-checker pipeline
    council     Multi-model council with chairman arbiter
    evaluate    Compute chrF++ and terminology accuracy on a predictions file
"""

import argparse
import json
import random
import sys
import time

import numpy as np
import torch

from mt_llm.eval import evaluate_translation
from mt_llm.io import get_terminology_fields, load_jsonl, load_terms_from_json
from mt_llm.pipeline import Pipeline


def _default_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return "auto" if torch.cuda.device_count() > 1 else "cuda"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _add_common_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default=_default_device())
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16", "auto"], default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--offline", action="store_true")


def cmd_translate(args: argparse.Namespace) -> int:
    """2-stage translate."""
    pipeline = Pipeline(
        mt_model_id=args.mt_model,
        checker_model_id=args.checker_model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        disable_checker=args.disable_checker,
        cache_dir=args.cache_dir,
        terminology_mode=args.terminology_mode,
    )
    pipeline.load_models()

    if args.text:
        terms = load_terms_from_json(args.terms) if args.terms else None
        result = pipeline.run(args.text, terms=terms, skip_checker=args.disable_checker)
        print(f"Source:  {result['source_text']}")
        print(f"MT:      {result['mt_translation']}")
        print(f"Final:   {result['final_translation']}")
        return 0

    return _run_batch(args, pipeline, mode="2stage")


def cmd_council(args: argparse.Namespace) -> int:
    """Council translate."""
    from mt_llm.council import Council

    council = Council(
        member_model_ids=args.members,
        chairman_model_id=args.chairman,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        cache_dir=args.cache_dir,
        peer_review=args.peer_review,
        terms_to_chairman=args.terms_to_chairman,
        allow_chairman_rewrite=args.allow_chairman_rewrite,
    )
    council.load_models()

    if args.text:
        terms = load_terms_from_json(args.terms) if args.terms else None
        result = council.run(args.text, terms=terms)
        print(f"Source:  {result['source_text']}")
        for i, cand in enumerate(result["candidates"]):
            print(f"Member{i}: {cand}")
        print(f"Final:   {result['final_translation']}")
        return 0

    return _run_batch(args, council, mode="council")


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Compute chrF++ + terminology accuracy on predictions vs references."""
    preds = load_jsonl(args.predictions)
    refs = load_jsonl(args.references)

    if len(preds) != len(refs):
        print(f"error: {len(preds)} predictions vs {len(refs)} references", file=sys.stderr)
        return 1

    chrf_total = 0.0
    term_total = 0
    term_found = 0
    n = 0

    for pred, ref in zip(preds, refs, strict=False):
        hypothesis = pred.get("prediction") or pred.get("translation") or pred.get("de", "")
        reference = ref.get("de", "")
        proper_terms, _ = get_terminology_fields(ref)
        if not hypothesis or not reference:
            continue
        ev = evaluate_translation(reference, hypothesis, proper_terms)
        chrf_total += ev["chrfpp_score"]
        if "proper_terms" in ev["terminology"]:
            term_total += ev["terminology"]["proper_terms"]["total_terms"]
            term_found += ev["terminology"]["proper_terms"]["found_terms"]
        n += 1

    if n == 0:
        print("error: no valid prediction/reference pairs", file=sys.stderr)
        return 1

    avg_chrf = chrf_total / n
    term_acc = (term_found / term_total) if term_total else float("nan")
    print(f"items:    {n}")
    print(f"chrF++:   {avg_chrf:.2f}")
    print(f"term_acc: {term_acc:.1%}" if term_total else "term_acc: n/a")
    return 0


def _run_batch(args, runner, mode: str) -> int:
    data = load_jsonl(args.data)
    if args.skip:
        data = data[args.skip :]
    if args.max_items:
        data = data[: args.max_items]

    sources = [e.get("en", "").strip() for e in data]
    refs = [e.get("de", "").strip() if e.get("de") else None for e in data]
    terms_list = [get_terminology_fields(e)[0] or None for e in data]

    t0 = time.time()
    if mode == "2stage":
        results = runner.run_batch(
            sources, terms_list=terms_list, skip_checker=args.disable_checker
        )
    else:  # council
        results = runner.run_batch(sources, terms_list=terms_list)
    dt = time.time() - t0

    if any(refs):
        chrfs = []
        for ref, res in zip(refs, results, strict=False):
            if ref is None:
                continue
            hyp = res.get("final_translation") or res.get("mt_translation", "")
            ev = evaluate_translation(ref, hyp, {})
            chrfs.append(ev["chrfpp_score"])
        if chrfs:
            print(f"items={len(chrfs)} avg_chrF++={sum(chrfs) / len(chrfs):.2f} time={dt:.1f}s")
    else:
        print(f"items={len(results)} time={dt:.1f}s")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for src, ref, res in zip(sources, refs, results, strict=False):
                row = {"en": src, "de": ref, "prediction": res.get("final_translation")}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {args.output}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mt-llm", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    # translate
    t = sub.add_parser("translate", help="2-stage MT + LLM-checker pipeline")
    t.add_argument("text", nargs="?", default=None, help="Single sentence to translate")
    t.add_argument("--data", type=str, default=None, help="JSONL file (batch mode)")
    t.add_argument("--max-items", type=int, default=None)
    t.add_argument("--skip", type=int, default=0)
    t.add_argument("--output", type=str, default=None, help="Write predictions JSONL")
    t.add_argument("--mt-model", type=str, default="facebook/nllb-200-3.3B")
    t.add_argument("--checker-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    t.add_argument("--disable-checker", action="store_true")
    t.add_argument("--terminology-mode", choices=["on", "off"], default="on")
    t.add_argument(
        "--terms", type=str, default=None, help="Terminology JSON path (single text mode)"
    )
    _add_common_model_args(t)
    t.set_defaults(func=cmd_translate)

    # council
    c = sub.add_parser("council", help="LLM Council with chairman arbiter")
    c.add_argument("text", nargs="?", default=None)
    c.add_argument("--data", type=str, default=None)
    c.add_argument("--max-items", type=int, default=None)
    c.add_argument("--skip", type=int, default=0)
    c.add_argument("--output", type=str, default=None)
    c.add_argument(
        "--members",
        nargs="+",
        default=[
            "togethercomputer/GPT-NeoXT-Chat-Base-20B",
            "ai-sage/GigaChat-20B-A3B-instruct",
            "ibm-granite/granite-20b-code-instruct-8k",
        ],
        help="Council member model IDs (default: report's 3 models)",
    )
    c.add_argument(
        "--chairman",
        type=str,
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="Chairman model ID (default: Mixtral-8x7B)",
    )
    c.add_argument(
        "--peer-review", action="store_true", help="Pass member peer reviews to chairman"
    )
    c.add_argument(
        "--terms-to-chairman", action="store_true", help="Pass terminology dictionary to chairman"
    )
    c.add_argument(
        "--allow-chairman-rewrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Chairman may rewrite if unsatisfied (default: on)",
    )
    c.add_argument("--terms", type=str, default=None)
    _add_common_model_args(c)
    c.set_defaults(func=cmd_council)

    # evaluate
    e = sub.add_parser("evaluate", help="chrF++ + terminology accuracy")
    e.add_argument("--predictions", type=str, required=True)
    e.add_argument("--references", type=str, required=True)
    e.set_defaults(func=cmd_evaluate)

    return p


def main(argv: list | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "seed"):
        _set_seed(args.seed)
    if getattr(args, "offline", False):
        import os

        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
