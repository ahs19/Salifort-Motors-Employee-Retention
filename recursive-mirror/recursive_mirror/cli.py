"""Command-line entry points. Real models are loaded only by the run command."""
from __future__ import annotations

import argparse
import sys

from .core import Config, canonical
from .demo import ScriptedTestDouble, demo_plan
from .runtime import HuggingFaceBackend, execute, load_checkpoint, load_plan


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Bounded recursive LLM strategy and self-prediction experiment")
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Deterministic test double only; not model or awareness evidence")
    demo.add_argument("--output", required=True, help="New run directory; existing directories are refused")
    run = commands.add_parser("run", help="Run a real local Hugging Face model on an explicit plan")
    run.add_argument("--plan", required=True)
    run.add_argument("--output", required=True, help="New run directory; existing directories are refused")
    run.add_argument("--model", required=True)
    run.add_argument("--revision", help="Full 40-character commit; required for remote downloads")
    run.add_argument("--allow-download", action="store_true", help="Explicitly permit downloading model/tokenizer files")
    run.add_argument("--device", choices=("cuda", "cpu", "auto"), default="auto")
    run.add_argument("--four-bit", action="store_true", help="Use NF4 quantization on CUDA")
    run.add_argument("--max-depth", type=int, default=4)
    run.add_argument("--max-calls", type=int, default=128)
    run.add_argument("--state", help="Hash-checked checkpoint carrying forward committed strategy and memory")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "demo":
            print("DEMO: deterministic scripted test double. Scores are software-test evidence only.", flush=True)
            model, plan, initial, config = ScriptedTestDouble(), demo_plan(), None, Config(max_depth=2, max_calls=64)
        else:
            plan = load_plan(args.plan)
            config = Config(max_depth=args.max_depth, max_calls=args.max_calls)
            model = HuggingFaceBackend(args.model, allow_download=args.allow_download,
                                       device=args.device, four_bit=args.four_bit,
                                       revision=args.revision)
            initial = load_checkpoint(args.state, expected_model_identity=model.identity) if args.state else None
        result = execute(model, plan, args.output, config, initial)
        print(canonical({"status": result.status, "calls": result.calls,
                         "model_identity": result.model_identity,
                         "transfer_improved": result.transfer_improved,
                         "committed_state_sha256": result.committed_state.sha256,
                         "output": args.output, "claim": result.claim}), flush=True)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
