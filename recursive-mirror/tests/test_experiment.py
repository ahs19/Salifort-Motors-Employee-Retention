"""Replicated experiments: per-seed runs, control arm, failure isolation and pooled analysis."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from recursive_mirror.cli import main
from recursive_mirror.core import Config, sections
from recursive_mirror.experiment import DEFAULT_CONTROL_STRATEGY, run_experiment, summarize
try:
    from test_tasks import solve          # unittest discover -s tests
except ImportError:  # pragma: no cover - run as a dotted module from the project root
    from tests.test_tasks import solve


class HeadroomDouble:
    """Deterministic stand-in: solves generated tasks only when told to work step by step."""
    identity = "headroom-test-double"
    deterministic = True

    def __init__(self, fail_on: str | None = None):
        self.fail_on = fail_on

    def complete(self, request):
        parts = sections(request.prompt)
        if self.fail_on and self.fail_on in request.prompt:
            raise ConnectionError("backend lost")
        if request.purpose == "forecast":
            return "70"
        if request.purpose == "answer":
            category, prompt = parts["TASK"].split("\n", 1)
            truth = solve(category.removeprefix("Category: "), prompt)
            careful = "step by step" in parts["STRATEGY"].lower()
            return f"Working.\nANSWER: {truth if careful or 'decimal' in category else truth + 'x'}"
        if request.purpose.startswith("perspective:"):
            return "Misses came from answering without intermediate steps."
        if request.purpose.startswith("revise_strategy:"):
            return "Work step by step: write every intermediate result, then check it."
        if request.purpose.startswith("revise_self_model:"):
            return "About 70% reliable on every task type."
        raise AssertionError(request.purpose)


SIZES = dict(rounds=2, discovery=3, validation=6, final=6)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_replicated_runs_pool_held_out_evidence_per_question(self):
        lines = []
        summary = run_experiment(HeadroomDouble(), self.root / "exp", seeds=(1, 2), config=Config(max_depth=2),
                                 log=lines.append, **SIZES)
        self.assertEqual([(r["seed"], r["arm"]) for r in summary["runs"]],
                         [("1", "control"), ("1", "self"), ("2", "control"), ("2", "self")])
        by_arm = {(r["seed"], r["arm"]): r for r in summary["runs"]}
        for seed in ("1", "2"):
            self.assertTrue(by_arm[(seed, "self")]["transfer_improved"])
            self.assertFalse(by_arm[(seed, "control")]["transfer_improved"])
            self.assertIn("step by step", by_arm[(seed, "self")]["committed_strategy"])
            self.assertEqual(by_arm[(seed, "control")]["committed_strategy"], DEFAULT_CONTROL_STRATEGY)
        pooled = summary["pooled"]
        self.assertEqual(pooled["self_vs_initial"]["n"], 12)
        self.assertGreater(pooled["self_vs_initial"]["accuracy_after"], pooled["self_vs_initial"]["accuracy_before"])
        self.assertLess(pooled["self_vs_initial"]["accuracy_p"], 0.01)
        self.assertEqual(pooled["self_vs_control"]["wins"], 0)
        self.assertEqual(pooled["self_vs_control"]["losses"], 0)
        self.assertEqual(pooled["control_self_improvement"]["accuracy_p"], 1.0)
        self.assertEqual(sum(row["n"] for row in summary["categories"].values()), 12)
        text = (self.root / "exp" / "summary.md").read_text(encoding="utf-8")
        self.assertIn("Selected policy vs its starting policy (self arm) | 12 |", text)
        self.assertEqual(json.loads((self.root / "exp" / "summary.json").read_text(encoding="utf-8")), summary)
        self.assertTrue(any("level 0: selected repair" in line for line in lines))
        # The same analysis can be recomputed from the saved files alone.
        self.assertEqual(summarize(self.root / "exp"), summary)

    def test_pooled_calibration_ignores_missing_forecasts(self):
        # Multi-agent review: counting a missing forecast as loss 1.0 made identical forecasts
        # look like a "significant" pooled calibration gain.
        from recursive_mirror.experiment import _paired
        def rows(missing):
            return {f"t{i}": {"correct": i % 2 == 0, "probability": None if i in missing else 0.9}
                    for i in range(24)}
        pooled = _paired(rows({1, 2, 3}), rows(set()))
        self.assertEqual((pooled["coverage_before"], pooled["coverage_after"]), (21 / 24, 1.0))
        self.assertEqual(pooled["paired_forecasts"], 21)
        self.assertEqual(pooled["brier_before"], pooled["brier_after"])
        self.assertEqual(pooled["brier_p"], 1.0)

    def test_total_time_budget_is_shared_out_as_it_is_spent(self):
        run_experiment(HeadroomDouble(), self.root / "exp", seeds=(1, 2), config=Config(max_depth=1),
                       log=lambda line: None, total_seconds=4000, **SIZES)
        budgets = [json.loads((self.root / "exp" / f"seed-{seed}" / arm / "result.json").read_text())["config"]["max_seconds"]
                   for seed in (1, 2) for arm in ("self", "control")]
        self.assertTrue(all(0 < b <= 4000 for b in budgets))
        self.assertGreaterEqual(budgets[-1], budgets[0])  # unspent time carries over to later runs

    def test_one_failed_run_does_not_discard_the_others(self):
        summary = run_experiment(HeadroomDouble(fail_on="Think step by step. Break"), self.root / "exp",
                                 seeds=(3,), config=Config(max_depth=1), log=lambda line: None, **SIZES)
        status = {r["arm"]: r["status"] for r in summary["runs"]}
        self.assertEqual(status["control"], "failed")
        self.assertIn(status["self"], ("depth_limit", "plan_exhausted"))
        failure = (self.root / "exp" / "seed-3" / "control" / "failure.txt").read_text(encoding="utf-8")
        self.assertIn("ConnectionError", failure)
        self.assertEqual(summary["pooled"]["self_vs_control"], {"n": 0})

    def test_experiment_refuses_existing_output_and_bad_seeds(self):
        (self.root / "taken").mkdir()
        with self.assertRaises(FileExistsError):
            run_experiment(HeadroomDouble(), self.root / "taken", seeds=(1,), config=Config(), **SIZES)
        for seeds in ((), (1, 1)):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                run_experiment(HeadroomDouble(), self.root / f"x{len(seeds)}", seeds=seeds, config=Config(), **SIZES)

    def test_cli_experiment_and_summarize(self):
        output = self.root / "cli-exp"
        stdout = io.StringIO()
        with patch("recursive_mirror.cli.HuggingFaceBackend", return_value=HeadroomDouble()), \
                redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            code = main(["experiment", "--output", str(output), "--model", "m", "--seeds", "4,5",
                         "--rounds", "1", "--discovery", "3", "--validation", "6", "--final", "6",
                         "--max-depth", "1", "--control-strategy", ""])
        self.assertEqual(code, 0, stdout.getvalue())
        self.assertEqual(sorted(p.name for p in output.glob("seed-*/*")), ["self", "self"])
        self.assertIn("# Recursive Mirror experiment summary", stdout.getvalue())
        self.assertTrue(re.search(r"\| 4 \| self \| \w+ \| 1/1 \|", stdout.getvalue()))
        with redirect_stdout(io.StringIO()) as again:
            self.assertEqual(main(["summarize", str(output)]), 0)
        self.assertIn("Pooled held-out comparisons", again.getvalue())
        with patch("recursive_mirror.cli.HuggingFaceBackend", return_value=HeadroomDouble()), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(main(["experiment", "--output", str(self.root / "bad"), "--model", "m",
                                   "--initial-strategy", "x"]), 1)
        self.assertIn("starts every run fresh", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
