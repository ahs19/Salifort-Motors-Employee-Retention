"""Command-line entry points. Real models are loaded only by the run command."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

from .core import Config, RecursiveMirror, State, canonical, stale_tasks
from .demo import ScriptedTestDouble, demo_plan
from .diagnostics import diagnose, diagnosis_markdown
from .experiment import DEFAULT_CONTROL_STRATEGY, run_experiment, summary_markdown, write_summary
from .runtime import DTYPES, HuggingFaceBackend, _save_exclusive, execute, load_checkpoint, load_plan
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
    make.add_argument("--exclude-state", help="Checkpoint whose already-used prompts the new plan must avoid")

    run = commands.add_parser("run", help="Run a real local Hugging Face model on an explicit plan")
    run.add_argument("--plan", required=True)
    experiment = commands.add_parser("experiment", help="Replicated runs over several generated plans, "
                                     "with an optional hand-written control start, and a pooled summary")
    experiment.add_argument("--seeds", type=_seeds, default=(2026, 2027, 2028))
    experiment.add_argument("--rounds", type=int, default=4)
    experiment.add_argument("--discovery", type=int, default=8)
    experiment.add_argument("--validation", type=int, default=24)
    experiment.add_argument("--final", type=int, default=48)
    experiment.add_argument("--categories", type=_categories, default=CATEGORIES)
    experiment.add_argument("--total-seconds", type=int, default=0,
                            help="Wall-clock budget shared by all runs as it is spent (0 = none)")
    experiment.add_argument("--control-strategy", default=DEFAULT_CONTROL_STRATEGY,
                            help="Hand-written starting strategy for the control arm ('' disables it)")
    summary = commands.add_parser("summarize", help="Recompute the pooled summary of an experiment directory")
    summary.add_argument("directory")
    health = commands.add_parser("diagnose", help="Protocol health of one run directory (works on partial runs)")
    health.add_argument("directory")
    health.add_argument("--strict", action="store_true", help="Exit 1 if any warning is raised")
    for sub in (run, experiment):
        _model_options(sub)
    return root


def _seeds(text: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part) for part in text.split(",") if part.strip())
    except ValueError:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from None
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise argparse.ArgumentTypeError("seeds must be distinct nonnegative integers")
    return seeds


def _model_options(run: argparse.ArgumentParser) -> None:
    run.add_argument("--output", required=True, help="New run directory; existing directories are refused")
    run.add_argument("--model", required=True)
    run.add_argument("--revision", help="Full 40-character commit; required for remote downloads")
    run.add_argument("--allow-download", action="store_true", help="Explicitly permit downloading model/tokenizer files")
    run.add_argument("--device", choices=("cuda", "cpu", "auto"), default="auto")
    run.add_argument("--four-bit", action="store_true", help="Use NF4 quantization on CUDA")
    run.add_argument("--dtype", choices=DTYPES, default="auto",
                     help="auto = float16 on CUDA (T4), float32 on CPU; bfloat16 suits L4/A100")
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
    run.add_argument("--batch-size", type=int, default=Config.batch_size,
                     help="Independent evaluation requests generated together on the GPU")
    run.add_argument("--max-seconds", type=int, default=0,
                     help="Wall-clock budget per run: stop recursing, still run the final test (0 = none)")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "make-plan":
            used = load_checkpoint(args.exclude_state) if args.exclude_state else State()
            plan = generate_plan(args.seed, args.rounds, args.discovery, args.validation, args.final,
                                 args.categories, exclude={o.prompt_sha256 for o in used.observations})
            stale = stale_tasks(plan, used)
            if stale:
                raise ValueError(f"task ID {stale[0]} was already used by that checkpoint; choose another --seed")
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            _save_exclusive(Path(args.output), asdict(plan))
            print(canonical({"plan": args.output, "rounds": len(plan.rounds),
                             "tasks": len(plan.tasks()), "categories": list(args.categories)}), flush=True)
            return 0
        if args.command == "diagnose":
            report = diagnose(args.directory)
            print(diagnosis_markdown(args.directory, report), flush=True)
            return 1 if args.strict and report["warnings"] else 0
        if args.command == "summarize":
            summary = write_summary(args.directory)
            print(summary_markdown(summary), flush=True)
            return 0
        if args.command == "demo":
            print("DEMO: deterministic scripted test double. Scores are software-test evidence only.", flush=True)
            model, plan, initial, config = ScriptedTestDouble(), demo_plan(), None, Config(max_depth=2, max_calls=200)
        else:
            config = Config(max_depth=args.max_depth, max_calls=args.max_calls, patience=args.patience,
                            candidate_kinds=args.candidates, answer_tokens=args.answer_tokens,
                            reveal_discovery_answers=not args.hide_discovery_answers, seed=args.seed,
                            max_p_value=args.max_p_value, batch_size=args.batch_size,
                            max_seconds=args.max_seconds)
            model = HuggingFaceBackend(args.model, allow_download=args.allow_download,
                                       device=args.device, four_bit=args.four_bit,
                                       revision=args.revision, max_input_tokens=args.max_input_tokens,
                                       dtype=args.dtype)
            if args.command == "experiment":
                if args.state or args.initial_strategy or args.initial_self_model:
                    raise ValueError("experiment starts every run fresh; use run for --state or --initial-*")
                summary = run_experiment(model, args.output, seeds=args.seeds, config=config,
                                         rounds=args.rounds, discovery=args.discovery,
                                         validation=args.validation, final=args.final,
                                         categories=args.categories,
                                         control_strategy=args.control_strategy or None,
                                         log=lambda line: print(line, flush=True),
                                         total_seconds=args.total_seconds)
                completed = sum("final_n" in run for run in summary["runs"])
                return 0 if completed == len(summary["runs"]) else 1
            plan = load_plan(args.plan)
            custom = {key: value for key, value in (("strategy", args.initial_strategy),
                                                    ("self_model", args.initial_self_model)) if value is not None}
            if custom and args.state:
                raise ValueError("--initial-strategy/--initial-self-model cannot be combined with --state")
            initial = (load_checkpoint(args.state, expected_model_identity=model.identity) if args.state
                       else State(**custom) if custom else None)
            stale = stale_tasks(plan, initial) if initial is not None else []
            if stale:  # refuse before any run directory is created
                raise ValueError(f"A continued run requires fresh task IDs and prompts; {stale[0]} was already "
                                 "used (make-plan --exclude-state avoids this)")
        result = execute(model, plan, args.output, config, initial)
        print(canonical({"status": result.status, "calls": result.calls, "cache_hits": result.cache_hits,
                         "model_identity": result.model_identity,
                         "accepted_levels": sum(s.accepted for s in result.steps),
                         "transfer_improved": result.transfer_improved,
                         "final_reason": result.final_comparison.reason,
                         "initial_final_metrics": asdict(result.initial_final_metrics),
                         "candidate_final_metrics": asdict(result.candidate_final_metrics),
                         "committed_state_sha256": result.committed_state.sha256,
                         "output": args.output, "claim": result.claim}), flush=True)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
