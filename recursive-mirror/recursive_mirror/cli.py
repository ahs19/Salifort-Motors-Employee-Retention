"""Command-line entry points. Real models are loaded only by the run command."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

from .core import Config, RecursiveMirror, State, canonical
from .demo import ScriptedTestDouble, demo_plan
from .runtime import HuggingFaceBackend, _save_exclusive, execute, load_checkpoint, load_plan
from .tasks import CATEGORIES, generate_plan


def _kinds(text: str) -> tuple[str, ...]:
    kinds = tuple(k.strip() for k in text.split(",") if k.strip())
    if not kinds or any(k not in RecursiveMirror.KINDS for k in kinds):
        raise argparse.ArgumentTypeError(f"choose a comma-separated subset of {','.join(RecursiveMirror.KINDS)}")
    return kinds


def _categories(text: str) -> tuple[str, ...]:
    names = tuple(c.strip() for c in text.split(",") if c.strip())
    if not names or any(c not in CATEGORIES for c in names):
        raise argparse.ArgumentTypeError(f"choose a comma-separated subset of {','.join(CATEGORIES)}")
    return names


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Bounded recursive self-improvement of an LLM agent's "
                                               "strategy and self-prediction")
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Deterministic test double only; not model or awareness evidence")
    demo.add_argument("--output", required=True, help="New run directory; existing directories are refused")

    make = commands.add_parser("make-plan", help="Write a deterministic exact-answer task plan with headroom")
    make.add_argument("--output", required=True, help="New JSON file; existing files are refused")
    make.add_argument("--seed", type=int, default=2026)
    make.add_argument("--rounds", type=int, default=2)
    make.add_argument("--discovery", type=int, default=6)
    make.add_argument("--validation", type=int, default=10)
    make.add_argument("--final", type=int, default=16)
    make.add_argument("--categories", type=_categories, default=CATEGORIES)

    run = commands.add_parser("run", help="Run a real local Hugging Face model on an explicit plan")
    run.add_argument("--plan", required=True)
    run.add_argument("--output", required=True, help="New run directory; existing directories are refused")
    run.add_argument("--model", required=True)
    run.add_argument("--revision", help="Full 40-character commit; required for remote downloads")
    run.add_argument("--allow-download", action="store_true", help="Explicitly permit downloading model/tokenizer files")
    run.add_argument("--device", choices=("cuda", "cpu", "auto"), default="auto")
    run.add_argument("--four-bit", action="store_true", help="Use NF4 quantization on CUDA")
    run.add_argument("--max-input-tokens", type=int, default=8192,
                     help="Refuse (never truncate) longer prompts; v0.8 used 4096")
    run.add_argument("--max-depth", type=int, default=4)
    run.add_argument("--max-calls", type=int, default=400)
    run.add_argument("--patience", type=int, default=2)
    run.add_argument("--candidates", type=_kinds, default=Config.candidate_kinds,
                     help="Comma-separated proposal kinds per level: repair, rewrite, recalibrate")
    run.add_argument("--answer-tokens", type=int, default=Config.answer_tokens)
    run.add_argument("--max-p-value", type=float, default=Config.max_p_value,
                     help="Paired one-sided p-value bound every accepted gain must meet (1 disables it)")
    run.add_argument("--hide-discovery-answers", action="store_true",
                     help="Do not show reference answers for missed discovery tasks during reflection")
    run.add_argument("--seed", type=int, default=Config.seed)
    run.add_argument("--state", help="Hash-checked checkpoint carrying forward committed strategy and memory")
    run.add_argument("--initial-strategy", help="Start from this strategy instead of the default "
                     "(for example a hand-written control prompt); not combinable with --state")
    run.add_argument("--initial-self-model", help="Start from this self-model; not combinable with --state")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "make-plan":
            plan = generate_plan(args.seed, args.rounds, args.discovery, args.validation, args.final,
                                 args.categories)
            _save_exclusive(Path(args.output), asdict(plan))
            print(canonical({"plan": args.output, "rounds": len(plan.rounds),
                             "tasks": len(plan.tasks()), "categories": list(args.categories)}), flush=True)
            return 0
        if args.command == "demo":
            print("DEMO: deterministic scripted test double. Scores are software-test evidence only.", flush=True)
            model, plan, initial, config = ScriptedTestDouble(), demo_plan(), None, Config(max_depth=2, max_calls=200)
        else:
            plan = load_plan(args.plan)
            config = Config(max_depth=args.max_depth, max_calls=args.max_calls, patience=args.patience,
                            candidate_kinds=args.candidates, answer_tokens=args.answer_tokens,
                            reveal_discovery_answers=not args.hide_discovery_answers, seed=args.seed,
                            max_p_value=args.max_p_value)
            model = HuggingFaceBackend(args.model, allow_download=args.allow_download,
                                       device=args.device, four_bit=args.four_bit,
                                       revision=args.revision, max_input_tokens=args.max_input_tokens)
            custom = {key: value for key, value in (("strategy", args.initial_strategy),
                                                    ("self_model", args.initial_self_model)) if value is not None}
            if custom and args.state:
                raise ValueError("--initial-strategy/--initial-self-model cannot be combined with --state")
            initial = (load_checkpoint(args.state, expected_model_identity=model.identity) if args.state
                       else State(**custom) if custom else None)
        result = execute(model, plan, args.output, config, initial)
        print(canonical({"status": result.status, "calls": result.calls, "cache_hits": result.cache_hits,
                         "model_identity": result.model_identity,
                         "accepted_levels": sum(s.accepted for s in result.steps),
                         "transfer_improved": result.transfer_improved,
                         "initial_final_metrics": asdict(result.initial_final_metrics),
                         "candidate_final_metrics": asdict(result.candidate_final_metrics),
                         "committed_state_sha256": result.committed_state.sha256,
                         "output": args.output, "claim": result.claim}), flush=True)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
