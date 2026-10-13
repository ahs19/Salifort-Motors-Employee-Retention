"""Protocol-health diagnostics read from a run's journal (works on partial runs too)."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from .core import parse_forecast


def diagnose(run_dir: str | Path) -> dict:
    """Counts that show whether the model followed the protocol, and how fast it ran."""
    path = Path(run_dir) / "journal.jsonl"
    purposes: dict[int, str] = {}
    outcome = Counter()
    truncated = Counter()
    tokens = Counter()
    answers, forecasts, formats, accuracy = Counter(), Counter(), Counter(), {}
    elapsed, batch_sizes, events = 0.0, Counter(), Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line cut by an interrupted write
        kind = event.get("event")
        events[kind] += 1
        elapsed = max(elapsed, event.get("elapsed_seconds", 0.0))
        if kind == "llm_request":
            purpose = event["request"]["purpose"].split(":")[0]
            purposes[event["call"]] = purpose
            batch_sizes[event.get("batch_size", 1)] += 1
        elif kind in ("llm_response", "llm_truncated"):
            purpose = purposes.get(event["call"], "unknown")
            outcome[purpose] += 1
            if kind == "llm_truncated":
                truncated[purpose] += 1
            receipt = event.get("backend_receipt") or {}
            tokens[purpose] += receipt.get("output_tokens", 0) or 0
        elif kind == "observation":
            answers[event["answer_status"]] += 1
            forecasts["valid" if event["probability"] is not None else "invalid"] += 1
            raw = event["forecast_raw"]
            parsed = parse_forecast(raw) is not None
            formats["percent" if parsed and "%" in raw else "bare_number" if parsed else "other"] += 1
            key = (event["phase"], event["category"])
            hit = accuracy.setdefault(key, [0, 0])
            hit[0] += 1
            hit[1] += int(event["correct"])
    total_tokens = sum(tokens.values())
    return {"events": dict(events), "elapsed_seconds": round(elapsed, 1),
            "output_tokens": dict(tokens), "tokens_per_second": round(total_tokens / elapsed, 2) if elapsed else None,
            "calls_by_purpose": dict(outcome), "truncated_by_purpose": dict(truncated),
            "answer_status": dict(answers), "forecasts": dict(forecasts), "forecast_formats": dict(formats),
            "batch_sizes": dict(batch_sizes),
            "accuracy": {f"{phase}/{category}": {"n": n, "correct": c}
                         for (phase, category), (n, c) in sorted(accuracy.items())},
            "warnings": _warnings(outcome, truncated, answers, forecasts, formats)}


def _warnings(outcome: Counter, truncated: Counter, answers: Counter, forecasts: Counter,
              formats: Counter) -> list[str]:
    notes = []
    total_formats = sum(formats.values())
    if total_formats and formats["percent"] / total_formats < 0.95:
        notes.append(f"only {formats['percent']}/{total_formats} forecasts used the requested 'NN%' form "
                     "(bare numbers are invalid on integer-answer tasks)")
    for purpose, n in outcome.items():
        if n and truncated[purpose] / n > 0.25:
            notes.append(f"{truncated[purpose]}/{n} {purpose} replies hit the token limit")
    total_answers = sum(answers.values())
    if total_answers and answers["answer_line"] / total_answers < 0.5:
        notes.append(f"only {answers['answer_line']}/{total_answers} answers used the 'ANSWER:' line")
    total_forecasts = sum(forecasts.values())
    if total_forecasts and forecasts["invalid"] / total_forecasts > 0.2:
        notes.append(f"{forecasts['invalid']}/{total_forecasts} forecasts were not a 0-100 integer")
    return notes


def diagnosis_markdown(run_dir: str | Path, report: dict) -> str:
    lines = [f"# Diagnostics: {run_dir}", "",
             f"- Wall time: {report['elapsed_seconds']} s; output tokens/s: {report['tokens_per_second']}",
             f"- Calls by purpose: {report['calls_by_purpose']}",
             f"- Truncated by purpose: {report['truncated_by_purpose'] or 'none'}",
             f"- Output tokens by purpose: {report['output_tokens']}",
             f"- Answer extraction: {report['answer_status']}",
             f"- Forecasts: {report['forecasts']}; formats: {report['forecast_formats']}",
             f"- Batch sizes (requests): {report['batch_sizes']}", "",
             "| phase/category | n | correct |", "|---|---|---|"]
    lines += [f"| {key} | {row['n']} | {row['correct']} |" for key, row in report["accuracy"].items()]
    lines += ["", "Warnings: " + ("; ".join(report["warnings"]) if report["warnings"] else "none"), ""]
    return "\n".join(lines)
