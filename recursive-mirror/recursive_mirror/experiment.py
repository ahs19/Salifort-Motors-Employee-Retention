"""Replicated self-improvement experiments and their pooled held-out analysis.

One experiment runs the recursive loop on several independently generated task
plans (seeds). For each seed it can also run a control arm that starts from a
hand-written strategy. Every run keeps its own journal, result, checkpoint and
report. ``summarize`` reads only the saved result files, so the same analysis
can be repeated later on downloaded evidence.

Pooled questions, all measured on held-out final tasks that played no part in
any selection decision:

1. self_vs_initial: does the policy the loop selected beat the policy it
   started from?
2. self_vs_control: does it beat a hand-written step-by-step prompt on the
   same tasks?
3. control_self_improvement: can the loop improve that hand-written prompt?
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import traceback
from typing import Callable

from .core import Config, LanguageModel, State, canonical, paired_p_value
from .diagnostics import diagnose
from .runtime import _save_exclusive, execute, read_json
from .tasks import CATEGORIES, generate_plan


DEFAULT_CONTROL_STRATEGY = ("Think step by step. Break the task into small steps, write each intermediate "
                            "result, and check the result before the final answer line.")
SUMMARY_FORMAT = "recursive-mirror-experiment-summary-v1"


def run_experiment(model: LanguageModel, output: str | Path, *, seeds: tuple[int, ...], config: Config,
                   rounds: int = 3, discovery: int = 12, validation: int = 24, final: int = 48,
                   categories: tuple[str, ...] = CATEGORIES, control_strategy: str | None = DEFAULT_CONTROL_STRATEGY,
                   log: Callable[[str], None] = print, total_seconds: int = 0) -> dict:
    """Run every seed (and control) in order; one failed run never discards the others.

    total_seconds, when set, is shared out as it is spent: each run gets the
    remaining time divided by the runs still to come, so time a fast run saves
    carries over and a slow run cannot push the last run past the deadline.
    """
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list of distinct integers")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    _save_exclusive(root / "experiment.json", {
        "format": "recursive-mirror-experiment-v1", "model_identity": model.identity, "seeds": list(seeds),
        "plan": {"rounds": rounds, "discovery": discovery, "validation": validation, "final": final,
                 "categories": list(categories)},
        "control_strategy": control_strategy, "config": asdict(config)})
    arms = [("self", None)] + ([("control", State(strategy=control_strategy))] if control_strategy else [])
    started, runs_left = time.monotonic(), len(seeds) * len(arms)
    for seed in seeds:
        plan = generate_plan(seed, rounds, discovery, validation, final, categories)
        for arm, initial in arms:
            directory = root / f"seed-{seed}" / arm
            log(f"[seed {seed} | {arm}] starting: {len(plan.tasks())} tasks, budget {config.max_calls} calls")

            def progress(event: dict, seed=seed, arm=arm) -> None:
                if event.get("event") == "selection":
                    chosen = event.get("selected_kind") or "none"
                    log(f"[seed {seed} | {arm}] level {event['round_index']}: selected {chosen}; "
                        + ", ".join(f"{c['kind']}={c['reason']}" for c in event["candidates"]))

            run_config = config
            if total_seconds:
                share = int((total_seconds - (time.monotonic() - started)) / runs_left)
                run_config = replace(config, max_seconds=max(60, share))
            runs_left -= 1
            try:
                result = execute(model, plan, directory, run_config, initial, on_event=progress)
                final_cmp = result.final_comparison
                health = diagnose(directory)
                log(f"[seed {seed} | {arm}] done: final accuracy {final_cmp.parent.accuracy:.0%} -> "
                    f"{final_cmp.candidate.accuracy:.0%}, transfer={result.transfer_improved}, calls={result.calls}, "
                    f"{health['elapsed_seconds']} s, {health['tokens_per_second']} tok/s"
                    + (f"; warnings: {'; '.join(health['warnings'])}" if health["warnings"] else ""))
            except Exception as exc:  # keep going: other runs are independent evidence
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
                log(f"[seed {seed} | {arm}] FAILED: {type(exc).__name__}: {exc}")
            write_summary(root)
    summary = write_summary(root)
    log(summary_markdown(summary))
    return summary


def _final_rows(result: dict, phase: str) -> dict[str, dict]:
    return {row["task_id"]: row for row in result["candidate_state"]["observations"] if row["phase"] == phase}


def _paired(left: dict[str, dict], right: dict[str, dict]) -> dict:
    """Paired comparison of right over left on shared task IDs.

    Calibration is measured the way the gate measures it: Brier loss only on
    tasks where both sides produced a valid forecast, with forecast coverage
    reported separately, so a missing forecast can never pose as a gain.
    """
    ids = sorted(left.keys() & right.keys())
    if not ids:
        return {"n": 0}

    def loss(row):
        return (row["probability"] - int(row["correct"])) ** 2

    both = [i for i in ids if left[i]["probability"] is not None and right[i]["probability"] is not None]
    acc_l = sum(left[i]["correct"] for i in ids) / len(ids)
    acc_r = sum(right[i]["correct"] for i in ids) / len(ids)
    wins = sum(right[i]["correct"] and not left[i]["correct"] for i in ids)
    losses = sum(left[i]["correct"] and not right[i]["correct"] for i in ids)
    return {"n": len(ids), "accuracy_before": acc_l, "accuracy_after": acc_r, "wins": wins, "losses": losses,
            "accuracy_p": paired_p_value([int(right[i]["correct"]) - int(left[i]["correct"]) for i in ids]),
            "coverage_before": sum(left[i]["probability"] is not None for i in ids) / len(ids),
            "coverage_after": sum(right[i]["probability"] is not None for i in ids) / len(ids),
            "paired_forecasts": len(both),
            "brier_before": sum(loss(left[i]) for i in both) / len(both) if both else None,
            "brier_after": sum(loss(right[i]) for i in both) / len(both) if both else None,
            "brier_p": paired_p_value([loss(left[i]) - loss(right[i]) for i in both])}


def summarize(root: str | Path) -> dict:
    """Pool every completed run under an experiment directory."""
    root = Path(root)
    runs, pools = [], {"self_vs_initial": ({}, {}), "self_vs_control": ({}, {}),
                       "control_self_improvement": ({}, {})}
    for seed_dir in sorted(root.glob("seed-*")):
        seed = seed_dir.name.removeprefix("seed-")
        finals = {}
        for arm_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
            arm = arm_dir.name
            if not (arm_dir / "result.json").is_file():
                runs.append({"seed": seed, "arm": arm, "status": "failed" if (arm_dir / "failure.txt").is_file()
                             else "incomplete"})
                continue
            result = read_json(arm_dir / "result.json")
            initial, candidate = _final_rows(result, "final_initial"), _final_rows(result, "final_candidate")
            finals[arm] = (initial, candidate)
            cmp = result["final_comparison"]
            runs.append({"seed": seed, "arm": arm, "status": result["status"], "calls": result["calls"],
                         "cache_hits": result["cache_hits"],
                         "accepted_levels": sum(step["accepted"] for step in result["steps"]),
                         "levels": len(result["steps"]), "transfer_improved": result["transfer_improved"],
                         "final_reason": cmp["reason"], "final_n": cmp["parent"]["n"],
                         "final_accuracy_initial": cmp["parent"]["accuracy"],
                         "final_accuracy_candidate": cmp["candidate"]["accuracy"],
                         "final_wins": cmp["wins"], "final_losses": cmp["losses"],
                         "committed_strategy": result["committed_state"]["strategy"],
                         "candidate_strategy": result["candidate_state"]["strategy"]})
        # Task IDs repeat across seeds' arms but not across seeds, so seed-prefixing keeps pairs exact.
        if "self" in finals:
            initial, candidate = finals["self"]
            pools["self_vs_initial"][0].update({f"{seed}:{k}": v for k, v in initial.items()})
            pools["self_vs_initial"][1].update({f"{seed}:{k}": v for k, v in candidate.items()})
        if "control" in finals:
            initial, candidate = finals["control"]
            pools["control_self_improvement"][0].update({f"{seed}:{k}": v for k, v in initial.items()})
            pools["control_self_improvement"][1].update({f"{seed}:{k}": v for k, v in candidate.items()})
            if "self" in finals:
                pools["self_vs_control"][0].update({f"{seed}:{k}": v for k, v in initial.items()})
                pools["self_vs_control"][1].update({f"{seed}:{k}": v for k, v in finals["self"][1].items()})
    pooled = {name: _paired(*pair) for name, pair in pools.items()}
    meta = read_json(root / "experiment.json") if (root / "experiment.json").is_file() else {}
    return {"format": SUMMARY_FORMAT, "model_identity": meta.get("model_identity"), "runs": runs,
            "pooled": pooled, "categories": _category_table(root)}


def _category_table(root: Path) -> dict:
    """Held-out accuracy per task type: initial vs selected policy, self arm pooled over seeds."""
    table: dict[str, list[int]] = {}
    for path in sorted(root.glob("seed-*/self/result.json")):
        result = read_json(path)
        initial, candidate = _final_rows(result, "final_initial"), _final_rows(result, "final_candidate")
        for task_id, row in initial.items():
            if task_id in candidate:
                counts = table.setdefault(row["category"], [0, 0, 0])
                counts[0] += 1
                counts[1] += int(row["correct"])
                counts[2] += int(candidate[task_id]["correct"])
    return {category: {"n": n, "accuracy_initial": a / n, "accuracy_selected": b / n}
            for category, (n, a, b) in sorted(table.items())}


def write_summary(root: str | Path) -> dict:
    root = Path(root)
    summary = summarize(root)
    for name, text in (("summary.json", canonical(summary) + "\n"), ("summary.md", summary_markdown(summary))):
        temporary = root / (name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(root / name)
    return summary


def _pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


def summary_markdown(summary: dict) -> str:
    lines = ["# Recursive Mirror experiment summary", "", f"Model: `{summary.get('model_identity')}`", "",
             "## Runs", "", "| seed | arm | status | levels accepted | final accuracy (initial -> selected) | "
             "wins/losses | committed |", "|---|---|---|---|---|---|---|"]
    for run in summary["runs"]:
        if "final_n" not in run:
            lines.append(f"| {run['seed']} | {run['arm']} | {run['status']} | | | | |")
            continue
        lines.append(f"| {run['seed']} | {run['arm']} | {run['status']} | {run['accepted_levels']}/{run['levels']} | "
                     f"{_pct(run['final_accuracy_initial'])} -> {_pct(run['final_accuracy_candidate'])} "
                     f"(n={run['final_n']}) | {run['final_wins']}/{run['final_losses']} | "
                     f"{'yes' if run['transfer_improved'] else 'no'} |")
    titles = {"self_vs_initial": "Selected policy vs its starting policy (self arm)",
              "self_vs_control": "Self-improved policy vs hand-written step-by-step prompt",
              "control_self_improvement": "Selected policy vs the hand-written starting prompt (control arm)"}
    lines += ["", "## Pooled held-out comparisons", "",
              "| comparison | tasks | accuracy | wins/losses | p (accuracy) | forecast coverage | "
              "Brier on paired valid forecasts | p (Brier) |", "|---|---|---|---|---|---|---|---|"]
    for key, title in titles.items():
        p = summary["pooled"][key]
        if not p.get("n"):
            lines.append(f"| {title} | 0 | | | | | | |")
            continue
        brier = ("n/a" if p["brier_before"] is None else
                 f"{p['brier_before']:.3f} -> {p['brier_after']:.3f} (n={p['paired_forecasts']})")
        lines.append(f"| {title} | {p['n']} | {_pct(p['accuracy_before'])} -> {_pct(p['accuracy_after'])} | "
                     f"{p['wins']}/{p['losses']} | {p['accuracy_p']:.3f} | {_pct(p['coverage_before'])} -> "
                     f"{_pct(p['coverage_after'])} | {brier} | {p['brier_p']:.3f} |")
    if summary.get("categories"):
        lines += ["", "## Held-out accuracy by task type (self arm, pooled)", "",
                  "| task type | n | initial | selected |", "|---|---|---|---|"]
        for category, row in summary["categories"].items():
            lines.append(f"| {category} | {row['n']} | {_pct(row['accuracy_initial'])} | {_pct(row['accuracy_selected'])} |")
    lines += ["", "p-values are one-sided paired sign-flip tests over pooled held-out tasks (exploratory; "
              "no correction for the three comparisons). Final tasks never influenced any selection. "
              "Nothing here measures awareness.", ""]
    return "\n".join(lines)
