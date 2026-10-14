"""Independent behavioural tests; no model weights or network are required."""
from __future__ import annotations

from dataclasses import replace
import inspect
import re
import unittest
import unittest.mock

from recursive_mirror.core import (
    Config, Metrics, Observation, OutputTruncated, Plan, RecursiveMirror, Request, Round, State, Task, render,
    compare, digest, extract_answer, normalize_answer, paired_p_value, parse_forecast, sections, summarize,
)


def make_plan(rounds=3, final_count=1, expected="ok"):
    def task(name):
        return Task(name, f"Solve the distinct item {name}.", expected)

    return Plan(
        tuple(Round((task(f"discovery-{i}"),), (task(f"validation-{i}"),))
              for i in range(rounds)),
        tuple(task(f"final-{i}") for i in range(final_count)),
    )


def task_id(request):
    match = re.search(r"Solve the distinct item (\S+)\.", sections(request.prompt).get("TASK", ""))
    return match.group(1) if match else None


def level(request):
    return int(re.search(r"Level (\d+) of", sections(request.prompt)["FRAME"]).group(1)) - 1


def strategy_of(parts):
    return parts.get("STRATEGY", parts.get("STRATEGY YOU WILL USE"))


class FakeModel:
    """Returns deterministic text and records only the public request boundary."""

    identity = "deterministic-test-model"
    views = {
        "actor": "From the actor angle, this is one reasoning agent.",
        "reviewer": "From the reviewer angle, this is two distinct reasoning roles.",
        "whole_system": "Both descriptions can apply when their viewpoints differ.",
    }

    def __init__(self, evaluation=None, override=None, deterministic=False):
        self.requests = []
        self.live_recursion_depths = []
        self.evaluation = evaluation or self.improving_evaluation
        self.override = override
        self.deterministic = deterministic

    @staticmethod
    def improving_evaluation(request, parts):
        """Strategy k solves the tasks of level k-1 and earlier; final tasks need any revision."""
        version = int(strategy_of(parts).removeprefix("strategy-"))
        if request.purpose == "answer":
            match = re.search(r"-(\d+)$", task_id(request) or "")
            index = int(match.group(1)) if match and not task_id(request).startswith("final-") else 0
            return "Reasoning.\nANSWER: ok" if version > index else "Reasoning.\nANSWER: no"
        return str([10, 50, 80, 100][min(version, 3)])

    def complete(self, request):
        self.requests.append(request)
        # Observe the live call boundary, rather than merely trusting journal
        # labels that could also have been emitted by an iterative loop.
        self.live_recursion_depths.append(sum(
            frame.function == "_recurse"
            and frame.filename.replace("\\", "/").endswith("recursive_mirror/core.py")
            for frame in inspect.stack()
        ))
        if self.override is not None:
            response = self.override(request)
            if response is not None:
                return response
        parts = sections(request.prompt)
        if request.purpose in ("forecast", "answer"):
            return self.evaluation(request, parts)
        if request.purpose.startswith("perspective:"):
            return self.views[request.purpose.split(":", 1)[1]]
        if request.purpose.startswith("revise_strategy:"):
            return f"strategy-{level(request) + 1}"
        if request.purpose.startswith("revise_self_model:"):
            return f"self-model-{level(request) + 1}-{request.purpose.split(':')[1]}"
        raise AssertionError(f"Unexpected model purpose: {request.purpose}")


INITIAL = State("strategy-0", "self-model-0")


def C(**kwargs):
    """Config for single-item mechanics tests: one task can never reach p <= 0.25,
    so these tests switch the evidence bound off (EvidenceGateTests cover it). They
    exercise whole-strategy proposals; RuleProposalTests cover the add_rule kinds."""
    kwargs.setdefault("max_p_value", 1.0)
    kwargs.setdefault("candidate_kinds", ("repair", "recalibrate"))
    return Config(**kwargs)


def run_case(model=None, plan=None, config=None, initial=INITIAL):
    model = model or FakeModel()
    events = []
    engine = RecursiveMirror(model, plan or make_plan(), config or C(), events.append)
    return engine.run(initial), model, events, engine


class RecursiveExecutionTests(unittest.TestCase):
    def test_calls_are_genuinely_nested_and_updated_state_reaches_next_level(self):
        result, model, events, _ = run_case()
        discovery = [(strategy_of(sections(req.prompt)), depth)
                     for req, depth in zip(model.requests, model.live_recursion_depths)
                     if req.purpose == "forecast" and task_id(req).startswith("discovery-")]
        self.assertEqual(discovery, [("strategy-0", 1), ("strategy-1", 2), ("strategy-2", 3)])
        nesting = [(e["event"], e["round_index"]) for e in events
                   if e["event"] in ("recursive_enter", "recursive_return")]
        self.assertEqual(nesting, [
            ("recursive_enter", 0), ("recursive_enter", 1), ("recursive_enter", 2),
            ("recursive_return", 2), ("recursive_return", 1), ("recursive_return", 0),
        ])
        self.assertEqual(result.status, "plan_exhausted")
        # Per level: 2 discovery + 3 views + 2 proposal calls (one per kind) + 2 parent + 2x2 candidates.
        self.assertEqual(result.calls, 3 * 13 + 4)
        self.assertEqual(result.cache_hits, 0)
        self.assertTrue(all(step.accepted for step in result.steps))
        self.assertEqual([s.selected_kind for s in result.steps], ["repair"] * 3)
        self.assertTrue(result.transfer_improved)
        self.assertEqual(result.committed_state.strategy, "strategy-3")
        self.assertEqual(result.committed_state.version, 3)
        self.assertEqual(result.claim, "measured_agent_strategy_and_self_prediction_only")

    def test_depth_limit_bounds_nested_calls(self):
        result, model, _, _ = run_case(plan=make_plan(4), config=C(max_depth=2))
        self.assertEqual(result.status, "depth_limit")
        self.assertEqual(len(result.steps), 2)
        self.assertEqual(max(model.live_recursion_depths), 2)

    def test_plateau_stops_after_declared_number_of_rejections(self):
        model = FakeModel(evaluation=lambda req, parts: "ANSWER: ok" if req.purpose == "answer" else "50")
        result, _, _, _ = run_case(model, make_plan(5), C(max_depth=5, patience=2))
        self.assertEqual(result.status, "no_measured_progress")
        self.assertEqual(len(result.steps), 2)
        self.assertFalse(any(step.accepted for step in result.steps))
        self.assertFalse(result.transfer_improved)
        self.assertEqual(result.committed_state.strategy, INITIAL.strategy)

    def test_every_proposal_is_frozen_before_any_validation_request(self):
        result, model, events, _ = run_case(plan=make_plan(1))
        proposals = [i for i, e in enumerate(events) if e["event"] == "proposal"]
        self.assertEqual(len(proposals), 2)
        validation = [i for i, e in enumerate(events) if e["event"] == "llm_request"
                      and e["request"]["purpose"] in ("forecast", "answer")
                      and "validation-0" in e["request"]["prompt"]]
        self.assertEqual(len(validation), 6)
        self.assertLess(max(proposals), min(validation))
        by_kind = {events[i]["kind"]: events[i] for i in proposals}
        candidates = {c.kind: c for c in result.steps[0].candidates}
        for kind, event in by_kind.items():
            self.assertEqual(event["state_sha256"], candidates[kind].proposed_state_sha256)
            self.assertEqual(event["policy_sha256"], digest({"strategy": event["strategy"],
                                                             "self_model": event["self_model"]}))
        for request in model.requests:
            if request.purpose.startswith(("perspective:", "revise_")):
                self.assertNotIn("validation-0", request.prompt)
                self.assertNotIn("final-0", request.prompt)

    def test_reference_answers_reach_reflection_only_for_missed_discovery_tasks(self):
        def secret_plan():
            def task(name):
                return Task(name, f"Solve the distinct item {name}.", f"KEY_{name}_517")
            return Plan(tuple(Round((task(f"discovery-{i}"),), (task(f"validation-{i}"),))
                              for i in range(2)), (task("final-0"),))

        guess = lambda req, parts: "50" if req.purpose == "forecast" else "ANSWER: observed guess"  # noqa: E731
        for reveal in (True, False):
            with self.subTest(reveal=reveal):
                plan = secret_plan()
                _, model, _, _ = run_case(FakeModel(guess), plan, C(reveal_discovery_answers=reveal))
                hidden = [t for r in plan.rounds for t in r.validation] + list(plan.final_tasks)
                for request in model.requests:
                    text = request.system + request.prompt
                    for task in hidden:
                        self.assertNotIn(task.expected, text)
                    if request.purpose in ("forecast", "answer"):
                        self.assertNotIn("KEY_", text)
                reflections = [r for r in model.requests if r.purpose.startswith(("perspective:", "revise_"))]
                self.assertEqual(len(reflections), 10)
                for request in reflections:
                    key = plan.rounds[level(request)].discovery[0].expected
                    self.assertEqual(f"Reference answer: {key}" in request.prompt, reveal)
                    self.assertIn("Agent's final answer: observed guess", request.prompt)
                    other = plan.rounds[1 - level(request)].discovery[0].expected
                    self.assertNotIn(other, request.prompt)

    def test_answer_requests_carry_strategy_but_never_the_self_model(self):
        _, model, _, _ = run_case(plan=make_plan(1))
        answers = [r for r in model.requests if r.purpose == "answer"]
        forecasts = [r for r in model.requests if r.purpose == "forecast"]
        self.assertTrue(answers and forecasts)
        for request in answers:
            parts = sections(request.prompt)
            self.assertEqual(set(parts), {"STRATEGY", "TASK", "OUTPUT RULE"})
            self.assertNotIn("self-model", request.prompt)
            self.assertIn("ANSWER: <answer>", request.system)
        for request in forecasts:
            self.assertEqual(list(sections(request.prompt)),
                             ["STRATEGY YOU WILL USE", "SELF-MODEL", "TASK", "QUESTION"])

    def test_same_validation_item_uses_paired_seeds(self):
        _, model, _, _ = run_case(plan=make_plan(1))
        for purpose in ("forecast", "answer"):
            requests = [r for r in model.requests if r.purpose == purpose and task_id(r) == "validation-0"]
            self.assertEqual(len(requests), 3)
            self.assertEqual(len({r.seed for r in requests}), 1)
            self.assertEqual(len({sections(r.prompt)["TASK"] for r in requests}), 1)

    def test_final_heldout_failure_rolls_back_only_the_policy(self):
        def evaluation(req, parts):
            candidate = strategy_of(parts) != INITIAL.strategy
            final = task_id(req).startswith("final-")
            if req.purpose == "forecast":
                return "100"
            return "ANSWER: ok" if candidate != final else "ANSWER: incorrect"

        result, _, events, _ = run_case(FakeModel(evaluation), make_plan(1))
        self.assertTrue(result.steps[0].accepted)
        self.assertFalse(result.transfer_improved)
        self.assertEqual(result.candidate_state.strategy, "strategy-1")
        self.assertEqual(result.committed_state.strategy, INITIAL.strategy)
        self.assertEqual(result.committed_state.self_model, INITIAL.self_model)
        self.assertEqual(result.committed_state.version, INITIAL.version)
        self.assertEqual(len(result.committed_state.perspectives), 3)
        self.assertEqual(len(result.committed_state.observations), 1 + 3 + 2)
        self.assertEqual(result.initial_final_metrics.accuracy, 1.0)
        self.assertEqual(result.candidate_final_metrics.accuracy, 0.0)
        final = result.committed_state.attempts[-1]
        self.assertEqual((final.kind, final.round_index, final.outcome, final.reason),
                         ("final_transfer", -1, "rejected", "accuracy_regression"))
        self.assertEqual(final.strategy, "strategy-1")
        self.assertEqual(events[-1]["event"], "run_completed")

    def test_rejected_revisions_retain_perspectives_and_attempts(self):
        model = FakeModel(evaluation=lambda req, parts: "ANSWER: ok" if req.purpose == "answer" else "50")
        result, _, _, _ = run_case(model, make_plan(2))
        self.assertTrue(all(not step.accepted for step in result.steps))
        self.assertEqual(result.candidate_state.version, 0)
        self.assertEqual(result.candidate_state.strategy, INITIAL.strategy)
        state = result.committed_state
        self.assertEqual([p.text for p in state.perspectives], list(FakeModel.views.values()) * 2)
        self.assertEqual([p.round_index for p in state.perspectives], [0, 0, 0, 1, 1, 1])
        self.assertEqual(len(state.observations), 2 * (1 + 3) + 2)
        self.assertEqual([(a.round_index, a.kind, a.outcome, a.reason) for a in state.attempts], [
            (0, "repair", "rejected", "no_declared_gain"), (0, "recalibrate", "rejected", "no_declared_gain"),
            (1, "repair", "rejected", "no_declared_gain"), (1, "recalibrate", "rejected", "no_declared_gain"),
        ])
        self.assertEqual(state.attempts[0].strategy, "strategy-1")

    def test_later_reflection_sees_previous_attempts_and_views_once(self):
        model = FakeModel(evaluation=lambda req, parts: "ANSWER: ok" if req.purpose == "answer" else "50")
        run_case(model, make_plan(3), C(max_depth=3, patience=3))
        reflections = [r for r in model.requests if r.purpose == "perspective:actor"]
        self.assertEqual(len(reflections), 3)
        history = [sections(r.prompt)["PREVIOUS IMPROVEMENT ATTEMPTS"] for r in reflections]
        self.assertEqual(history[0], "(none yet)")
        self.assertIn("level 0 | repair | REJECTED (no_declared_gain)", history[1])
        self.assertIn("strategy tried: strategy-1", history[1])
        self.assertIn("strategy tried: strategy-2", history[2])
        for text in FakeModel.views.values():
            self.assertNotIn(text, reflections[0].prompt)
            self.assertEqual(reflections[1].prompt.count(text), 1)
            self.assertEqual(reflections[2].prompt.count(text), 1)

    def test_whole_system_view_integrates_the_other_angles_of_its_level(self):
        _, model, _, _ = run_case(plan=make_plan(1))
        whole = next(r for r in model.requests if r.purpose == "perspective:whole_system")
        other = sections(whole.prompt)["THIS LEVEL'S OTHER VIEWS"]
        self.assertIn(FakeModel.views["actor"], other)
        self.assertIn(FakeModel.views["reviewer"], other)

    def test_best_passing_candidate_is_selected_and_others_recorded(self):
        def evaluation(req, parts):
            strategy = strategy_of(parts)
            if req.purpose == "forecast":
                return "50"
            good = strategy == "rewrite-plan" or (strategy == "strategy-1" and task_id(req) != "v-b")
            return "ANSWER: ok" if good else "ANSWER: no"

        def override(req):
            if req.purpose == "revise_strategy:rewrite":
                return "rewrite-plan"
            return None

        plan = Plan((Round((Task("d", "Solve the distinct item d.", "ok"),),
                           (Task("v-a", "Solve the distinct item v-a.", "ok"),
                            Task("v-b", "Solve the distinct item v-b.", "ok"))),),
                    (Task("f", "Solve the distinct item f.", "ok"),))
        result, _, _, _ = run_case(FakeModel(evaluation, override), plan,
                                   C(candidate_kinds=("repair", "rewrite")))
        step = result.steps[0]
        self.assertEqual(step.selected_kind, "rewrite")
        self.assertEqual([(c.kind, c.comparison.candidate.accuracy) for c in step.candidates],
                         [("repair", 0.5), ("rewrite", 1.0)])
        self.assertEqual([(a.kind, a.outcome) for a in result.candidate_state.attempts[:2]],
                         [("repair", "passed_not_selected"), ("rewrite", "accepted")])
        self.assertEqual(result.candidate_state.strategy, "rewrite-plan")

    def test_unchanged_proposal_is_ineligible_and_costs_no_validation_calls(self):
        def override(req):
            if req.purpose.startswith("revise_strategy:"):
                return INITIAL.strategy
            if req.purpose.startswith("revise_self_model:"):
                return INITIAL.self_model
            return None

        result, model, _, _ = run_case(FakeModel(override=override), make_plan(1))
        self.assertEqual([(c.kind, c.eligible, c.reason) for c in result.steps[0].candidates],
                         [("repair", False, "unchanged"), ("recalibrate", False, "unchanged")])
        validation = [r for r in model.requests if task_id(r) == "validation-0"]
        self.assertEqual(len(validation), 2)
        self.assertEqual(result.calls, 2 + 3 + 2 + 2 + 4)

    def test_instance_is_one_use_after_success(self):
        _, model, events, engine = run_case(plan=make_plan(1))
        calls, event_count = len(model.requests), len(events)
        with self.assertRaisesRegex(RuntimeError, "only once"):
            engine.run(INITIAL)
        self.assertEqual(len(model.requests), calls)
        self.assertEqual(len(events), event_count)


class EvidenceRegressionTests(unittest.TestCase):
    """Failure modes observed in the 9 October 2026 v0.8 Colab trials."""

    def test_truncated_perspective_is_retained_and_run_continues(self):
        # v0.8 trial 1 aborted at call 5 when the whole-system view hit 384 tokens.
        def override(req):
            if req.purpose == "perspective:whole_system":
                raise OutputTruncated("token limit", "A long whole-system view that never")
            return None

        result, _, events, _ = run_case(FakeModel(override=override), make_plan(1))
        view = result.committed_state.perspectives[-1]
        self.assertEqual((view.angle, view.complete, view.text),
                         ("whole_system", False, "A long whole-system view that never"))
        self.assertTrue(any(e["event"] == "llm_truncated" for e in events))
        self.assertEqual(events[-1]["event"], "run_completed")

    def test_truncated_answer_is_scored_incorrect_without_aborting(self):
        def override(req):
            if req.purpose == "answer" and task_id(req) == "discovery-0":
                raise OutputTruncated("token limit", "Step 1: ... ANSWER: ok and then more")
            return None

        result, _, events, _ = run_case(FakeModel(override=override), make_plan(1))
        row = next(o for o in result.committed_state.observations if o.task_id == "discovery-0")
        self.assertEqual((row.answer, row.answer_status, row.correct), ("", "truncated", False))
        self.assertEqual(events[-1]["event"], "run_completed")

    def test_unusable_or_overlong_proposals_are_ineligible(self):
        cases = {
            "truncated": (lambda req: (_ for _ in ()).throw(OutputTruncated("limit", "Propose expanding"))
                          if req.purpose == "revise_strategy:repair" else None, "strategy_truncated"),
            "empty": (lambda req: "   " if req.purpose == "revise_strategy:repair" else None, "strategy_empty"),
            "long": (lambda req: "x" * 2401 if req.purpose == "revise_strategy:repair" else None,
                     "strategy_too_long"),
        }
        for name, (override, reason) in cases.items():
            with self.subTest(name=name):
                result, _, _, _ = run_case(FakeModel(override=override), make_plan(1))
                repair = result.steps[0].candidates[0]
                self.assertEqual((repair.kind, repair.eligible, repair.reason), ("repair", False, reason))
                attempt = result.committed_state.attempts[0]
                self.assertEqual((attempt.outcome, attempt.reason), ("ineligible", reason))

    def test_each_kind_moves_one_lever(self):
        # v0.8's real self-model replies ran to 1541 characters; a strategy proposal must not
        # depend on one. Strategy kinds keep the parent's self-model; only recalibrate changes it.
        for reply in ("", "y" * 2401):
            with self.subTest(length=len(reply)):
                override = lambda req: reply if req.purpose.startswith("revise_self_model:") else None  # noqa: E731
                result, model, _, _ = run_case(FakeModel(override=override), make_plan(1))
                repair, recalibrate = result.steps[0].candidates
                self.assertTrue(repair.eligible)
                self.assertEqual((repair.strategy, repair.self_model), ("strategy-1", INITIAL.self_model))
                self.assertFalse(recalibrate.eligible)
                self.assertTrue(recalibrate.reason.startswith("self_model_"))
                self.assertEqual(recalibrate.strategy, INITIAL.strategy)
                self.assertEqual([r.purpose for r in model.requests if r.purpose.startswith("revise_")],
                                 ["revise_strategy:repair", "revise_self_model:recalibrate"])

    def test_non_executable_strategy_advice_is_screened_before_validation(self):
        override = lambda req: ("Propose expanding the training dataset and add a feedback loop."  # noqa: E731
                                if req.purpose == "revise_strategy:repair" else None)
        result, model, _, _ = run_case(FakeModel(override=override), make_plan(1))
        repair = result.steps[0].candidates[0]
        self.assertEqual((repair.eligible, repair.reason), (False, "strategy_non_executable"))

    def test_forecast_that_is_the_task_answer_is_not_a_probability(self):
        # v0.8 trial 2: the forecast call returned "216", the answer to 27 x 8.
        plan = Plan((Round((Task("d", "Count it.", "3"), Task("w", "Count those.", "3")),
                           (Task("v", "Count that.", "2"),)),),
                    (Task("f", "Name the day.", "Sunday"),))

        def evaluation(req, parts):
            if req.purpose == "answer":
                return "ANSWER: 3"
            return {"Count it.": "3", "Count those.": "5", "Count that.": "2%",
                    "Name the day.": "70"}[parts["TASK"].split("\n", 1)[1]]

        result, _, _, _ = run_case(FakeModel(evaluation), plan)
        by_task = {o.task_id: o for o in result.committed_state.observations}
        self.assertIsNone(by_task["d"].probability)          # bare "3" is the (correct) answer
        self.assertIsNone(by_task["w"].probability)          # bare "5" may be a wrong answer: also invalid
        self.assertEqual(by_task["v"].probability, 0.02)     # an explicit percentage is a forecast
        self.assertEqual(by_task["f"].probability, 0.7)      # a bare number cannot be a weekday answer

    def test_wall_clock_budget_stops_recursion_but_keeps_the_final_test(self):
        engine = RecursiveMirror(FakeModel(), make_plan(3), C(max_seconds=1))
        with unittest.mock.patch("recursive_mirror.core.time.monotonic", side_effect=[0.0, 0.5, 5.0, 5.0, 5.0]):
            result = engine.run(INITIAL)
        self.assertEqual(result.status, "time_budget")
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.final_comparison.parent.n, 1)

    def test_non_executable_screen_ignores_negated_sentences(self):
        from recursive_mirror.core import non_executable
        self.assertTrue(non_executable("Include a diverse set of problems in its training, then check."))
        self.assertTrue(non_executable("Propose expanding the training dataset."))
        self.assertFalse(non_executable("Do not rely on training data; multiply digit by digit."))
        self.assertFalse(non_executable("Use a feedback loop: re-check each partial product."))

    def test_untested_attempts_are_not_presented_as_rejected(self):
        override = lambda req: "x" * 2401 if req.purpose == "revise_strategy:repair" else None  # noqa: E731
        _, model, _, _ = run_case(FakeModel(override=override), make_plan(2), C(patience=3))
        later = [r for r in model.requests if r.purpose == "perspective:actor"][1]
        history = sections(later.prompt)["PREVIOUS IMPROVEMENT ATTEMPTS"]
        self.assertIn("repair | NOT TESTED (strategy_too_long)", history)

    def test_wall_clock_admission_reserves_the_level_and_the_final_at_the_measured_pace(self):
        engine = RecursiveMirror(FakeModel(), make_plan(3), C(max_seconds=100))
        # Level 0 is admitted at once; by level 1, 13 calls took 60 s (about 4.6 s per call), so a
        # 13-call level plus the 4-call final would need about 78 s more and exceed 100 s.
        with unittest.mock.patch("recursive_mirror.core.time.monotonic", side_effect=[0.0, 0.0, 60.0, 60.0]):
            result = engine.run(INITIAL)
        self.assertEqual((result.status, len(result.steps)), ("time_budget", 1))

    def test_parent_forecast_failure_cannot_create_candidate_calibration_gain(self):
        # v0.8 trial 2, level 1: the parent echoed "216" instead of a probability,
        # scored Brier 1.0, and an otherwise identical candidate was "accepted".
        def evaluation(req, parts):
            if req.purpose == "answer":
                return "ANSWER: ok"
            parent = strategy_of(parts) == INITIAL.strategy
            return "216" if parent and task_id(req).startswith("validation") else "100"

        result, _, _, _ = run_case(FakeModel(evaluation), make_plan(1))
        comparison = result.steps[0].candidates[0].comparison
        self.assertEqual(comparison.parent.forecast_coverage, 0.0)
        self.assertEqual(comparison.parent.brier, 1.0)
        self.assertEqual(comparison.candidate.brier, 0.0)
        self.assertEqual(comparison.paired_n, 0)
        self.assertFalse(comparison.passed)
        self.assertEqual(comparison.reason, "no_declared_gain")
        self.assertFalse(result.steps[0].accepted)

    def test_empty_and_oversize_responses_are_recorded_not_fatal(self):
        def override(req):
            if req.purpose == "forecast" and task_id(req) == "discovery-0":
                return ""
            if req.purpose == "answer" and task_id(req) == "discovery-0":
                return "x" * 12001
            return None

        result, _, events, _ = run_case(FakeModel(override=override), make_plan(1))
        row = next(o for o in result.committed_state.observations if o.task_id == "discovery-0")
        self.assertEqual((row.forecast_raw, row.probability), ("", None))
        self.assertEqual((row.answer, row.answer_status, row.correct), ("", "oversize", False))
        raw = [e for e in events if e["event"] == "llm_response" and e["call"] == 2]
        self.assertEqual(raw[0]["response"], "x" * 12001)
        self.assertEqual(events[-1]["event"], "run_completed")


class BudgetAndAuditTests(unittest.TestCase):
    def test_budget_must_cover_final_comparison(self):
        with self.assertRaisesRegex(ValueError, "reserved final"):
            RecursiveMirror(FakeModel(), make_plan(final_count=2), C(max_calls=7))

    def test_final_comparison_is_preserved_when_no_round_fits(self):
        model, events = FakeModel(), []
        result = RecursiveMirror(model, make_plan(), C(max_calls=4), events.append).run(INITIAL)
        self.assertEqual(result.status, "call_budget")
        self.assertEqual(result.calls, 4)
        self.assertEqual(result.steps, ())
        self.assertEqual([req.purpose for req in model.requests], ["forecast", "answer", "forecast", "answer"])
        self.assertFalse(any(e["event"] == "recursive_enter" for e in events))
        self.assertFalse(result.transfer_improved)

    def test_round_cannot_consume_final_reserve(self):
        model = FakeModel()
        result = RecursiveMirror(model, make_plan(3), C(max_calls=13 + 4 + 12)).run(INITIAL)
        self.assertEqual(result.status, "call_budget")
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.calls, 17)
        self.assertEqual([task_id(req) for req in model.requests[-4:]], ["final-0"] * 4)

    def test_exact_budget_allows_complete_round_and_final_test(self):
        result = RecursiveMirror(FakeModel(), make_plan(1), C(max_calls=17)).run(INITIAL)
        self.assertEqual(result.calls, 17)
        self.assertEqual(result.status, "plan_exhausted")
        self.assertTrue(result.transfer_improved)

    def test_backend_exception_aborts_with_failure_audit(self):
        def fail(request):
            raise ConnectionError("backend unavailable")

        events = []
        model = FakeModel(override=fail)
        with self.assertRaisesRegex(ConnectionError, "backend unavailable"):
            RecursiveMirror(model, make_plan(1), emit=events.append).run(INITIAL)
        self.assertEqual(len(model.requests), 1)
        self.assertEqual([e["event"] for e in events[-2:]], ["llm_error", "run_failed"])
        self.assertEqual(events[-2]["error_type"], "ConnectionError")
        self.assertFalse(any(e["event"] in ("selection", "run_completed") for e in events))

    def test_backend_failure_retains_partial_text_without_accepting_it(self):
        class InterruptedReply(RuntimeError):
            partial_output = "This reply stopped before completion"

        def fail(request):
            raise InterruptedReply("generation interrupted")

        events = []
        with self.assertRaises(InterruptedReply):
            RecursiveMirror(FakeModel(override=fail), make_plan(1), emit=events.append).run(INITIAL)
        error = next(e for e in events if e["event"] == "llm_error")
        self.assertEqual(error["partial_output"], InterruptedReply.partial_output)
        self.assertFalse(any(e["event"] in ("observation", "selection", "run_completed") for e in events))

    def test_non_text_response_aborts(self):
        events = []
        with self.assertRaises(TypeError):
            RecursiveMirror(FakeModel(override=lambda req: 42), make_plan(1), emit=events.append).run(INITIAL)
        self.assertEqual(events[-1]["event"], "run_failed")

    def test_backend_receipt_is_saved_alongside_each_raw_response(self):
        model = FakeModel()
        receipt = {"input_tokens": 42, "output_tokens": 1, "finish_reason": "eos"}
        model.last_receipt = receipt
        events = []
        RecursiveMirror(model, make_plan(1), C(max_calls=4), events.append).run(INITIAL)
        responses = [event for event in events if event["event"] == "llm_response"]
        self.assertEqual(len(responses), 4)
        self.assertTrue(all(event["backend_receipt"] == receipt for event in responses))
        self.assertEqual([event["response"] for event in responses],
                         ["10", "Reasoning.\nANSWER: no", "10", "Reasoning.\nANSWER: no"])

    def test_oversized_request_stops_before_backend_without_truncation(self):
        model, events = FakeModel(), []
        engine = RecursiveMirror(model, make_plan(1), C(max_request_chars=100), events.append)
        with self.assertRaisesRegex(ValueError, "no truncation"):
            engine.run(INITIAL)
        self.assertEqual(model.requests, [])
        self.assertEqual(events[-1]["event"], "run_failed")
        self.assertEqual(events[-1]["calls"], 0)


class DeterministicCacheTests(unittest.TestCase):
    def test_deterministic_backend_reuses_identical_requests(self):
        model = FakeModel(deterministic=True)
        result, _, events, _ = run_case(model, make_plan(1))
        # The recalibrate candidate keeps the parent's strategy, so its two answer
        # requests per validation item are byte-identical to the parent's.
        hits = [e for e in events if e["event"] == "llm_cache_hit"]
        self.assertEqual([e["purpose"] for e in hits], ["answer"])
        self.assertEqual(result.cache_hits, 1)
        self.assertEqual(result.calls, 17 - 1)
        self.assertEqual(len(model.requests), result.calls)
        parent = next(o for o in result.committed_state.observations if o.phase == "validation_parent")
        recal = next(o for o in result.committed_state.observations if o.arm == "recalibrate")
        self.assertEqual(parent.answer, recal.answer)

    def test_unchanged_policy_final_arm_is_free_for_deterministic_backends(self):
        model = FakeModel(evaluation=lambda req, parts: "ANSWER: ok" if req.purpose == "answer" else "50",
                          deterministic=True)
        result, _, _, _ = run_case(model, make_plan(1, final_count=3))
        self.assertFalse(result.steps[0].accepted)
        self.assertEqual(result.initial_final_metrics, result.candidate_final_metrics)
        self.assertFalse(result.transfer_improved)
        finals = [r for r in model.requests if (task_id(r) or "").startswith("final-")]
        self.assertEqual(len(finals), 6)

    def test_nondeterministic_backend_never_uses_cache(self):
        result, model, events, _ = run_case(FakeModel(deterministic=False), make_plan(1))
        self.assertEqual(result.cache_hits, 0)
        self.assertFalse(any(e["event"] == "llm_cache_hit" for e in events))
        self.assertEqual(len(model.requests), 17)


def rows(pairs):
    return tuple(Observation(0, "validation_parent", f"t{i}", "general",
                             str(round(p * 100)) if p is not None else "x", p, "a", c, "0" * 64)
                 for i, (p, c) in enumerate(pairs))


class BatchingModel(FakeModel):
    """FakeModel plus complete_batch(); records batch sizes and per-request receipts."""

    def __init__(self, *args, fail_batch=False, truncate=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.batches, self.fail_batch, self.truncate = [], fail_batch, truncate

    def complete_batch(self, requests):
        self.batches.append([r.purpose for r in requests])
        if self.fail_batch:
            raise RuntimeError("CUDA out of memory")
        outcomes = []
        for request in requests:
            if task_id(request) in self.truncate and request.purpose == "answer":
                self.requests.append(request)
                outcomes.append(OutputTruncated("limit", "partial"))
            else:
                outcomes.append(self.complete(request))
        self.last_receipts = [{"row": i} for i in range(len(requests))]
        return outcomes


def batch_plan():
    def task(name):
        return Task(name, f"Solve the distinct item {name}.", "ok")
    return Plan((Round(tuple(task(f"d{i}") for i in range(3)), tuple(task(f"v{i}") for i in range(5))),),
                tuple(task(f"f{i}") for i in range(4)))


class BatchingTests(unittest.TestCase):
    def test_independent_evaluation_calls_are_batched_but_journalled_one_by_one(self):
        model = BatchingModel()
        result, _, events, _ = run_case(model, batch_plan(), C(batch_size=4))
        sizes = [len(b) for b in model.batches]
        self.assertTrue(sizes and max(sizes) == 4)
        self.assertTrue(all(len(set(b)) == 1 for b in model.batches))  # one purpose per batch
        requests = [e for e in events if e["event"] == "llm_request"]
        outcomes = [e for e in events if e["event"] in ("llm_response", "llm_truncated")]
        self.assertEqual(len(requests), result.calls)
        self.assertEqual(sorted(e["call"] for e in outcomes), list(range(1, result.calls + 1)))
        batched = [e for e in events if e["event"] == "llm_response" and e["backend_receipt"]]
        self.assertTrue(all("row" in e["backend_receipt"] for e in batched))
        sequential = RecursiveMirror(FakeModel(), batch_plan(), C(batch_size=4)).run(INITIAL)
        self.assertEqual(result.calls, sequential.calls)
        self.assertEqual(result.committed_state, sequential.committed_state)

    def test_batch_size_one_uses_the_sequential_path(self):
        model = BatchingModel()
        run_case(model, batch_plan(), C(batch_size=1))
        self.assertEqual(model.batches, [])

    def test_identical_requests_in_one_batch_are_answered_once_when_deterministic(self):
        model = BatchingModel(deterministic=True)
        engine = RecursiveMirror(model, batch_plan(), C())
        request = Request("forecast", "s", render([("STRATEGY YOU WILL USE", "strategy-0")]), 4, 1)
        replies = engine._call_many([request, request, replace(request, seed=2)])
        self.assertEqual([r.call for r in replies], [1, 1, 2])
        self.assertEqual(engine.calls, 2)
        self.assertEqual(engine._call_many([request])[0].call, 1)
        self.assertEqual(engine.cache_hits, 1)

    def test_truncated_row_in_a_batch_is_scored_incorrect_without_aborting(self):
        model = BatchingModel(truncate={"v1"})
        result, _, events, _ = run_case(model, batch_plan(), C(batch_size=8))
        rows = [o for o in result.committed_state.observations if o.task_id == "v1"]
        self.assertTrue(rows and all(o.answer_status == "truncated" and not o.correct for o in rows))
        self.assertTrue(any(e["event"] == "llm_truncated" for e in events))
        self.assertEqual(events[-1]["event"], "run_completed")

    def test_batch_failure_is_audited_and_aborts(self):
        model, events = BatchingModel(fail_batch=True), []
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            RecursiveMirror(model, batch_plan(), C(batch_size=8), events.append).run(INITIAL)
        error = next(e for e in events if e["event"] == "llm_error")
        self.assertEqual(error["batch_calls"], [1, 2, 3])
        self.assertEqual(events[-1]["event"], "run_failed")

    def test_batch_never_exceeds_the_call_budget(self):
        engine = RecursiveMirror(BatchingModel(), batch_plan(), C(max_calls=17))
        engine.calls = 16
        requests = [Request("forecast", "s", f"p{i}", 4, i) for i in range(2)]
        with self.assertRaisesRegex(RuntimeError, "budget"):
            engine._call_many(requests)
        self.assertEqual(engine.calls, 16)


def two_type_plan():
    def task(name, category):
        return Task(name, f"Solve the distinct item {name}.", "ok", category)
    return Plan((Round((task("d-a1", "alpha"), task("d-b1", "beta"), task("d-b2", "beta")),
                       tuple(task(f"v-{c}{i}", c) for c in ("alpha", "beta") for i in range(3))),
                 Round((task("d-a2", "alpha"), task("d-b3", "beta")),
                       tuple(task(f"w-{c}{i}", c) for c in ("alpha", "beta") for i in range(3)))),
                tuple(task(f"f-{c}{i}", c) for c in ("alpha", "beta") for i in range(3)))


def beta_needs_a_rule(req, parts):
    """alpha is always solved; beta only once the strategy carries a beta rule."""
    if req.purpose == "forecast":
        return "60%"
    category = parts["TASK"].splitlines()[0].removeprefix("Category: ")
    solved = category == "alpha" or "beta-rule" in parts["STRATEGY"]
    return "ANSWER: ok" if solved else "ANSWER: no"


RULE_REPLIES = {"revise_strategy:add_rule": "1. For beta tasks, apply the beta-rule method and check it.",
                "revise_strategy:add_rule_2": "For alpha tasks, keep doing what works.",
                "revise_self_model:recalibrate": "alpha: about 8 in 10 correct (1 of 1 so far)"}


class RuleProposalTests(unittest.TestCase):
    """add_rule kinds: one rule for the weakest task types, appended to the strategy."""

    def run_rules(self, replies=None, **config):
        override = lambda req: (replies or RULE_REPLIES).get(req.purpose)  # noqa: E731
        config.setdefault("candidate_kinds", ("add_rule", "add_rule_2", "recalibrate"))
        config.setdefault("max_depth", 2)
        return run_case(FakeModel(beta_needs_a_rule, override), two_type_plan(), C(**config))

    def test_rule_targets_the_weakest_type_and_is_appended_to_the_strategy(self):
        result, model, _, _ = self.run_rules()
        step = result.steps[0]
        self.assertEqual(step.selected_kind, "add_rule")
        self.assertEqual(result.committed_state.strategy,
                         INITIAL.strategy + "\n- For beta tasks, apply the beta-rule method and check it.")
        self.assertTrue(result.transfer_improved)
        first = next(r for r in model.requests if r.purpose == "revise_strategy:add_rule")
        parts = sections(first.prompt)
        self.assertTrue(parts["TARGET TASK TYPE"].startswith("beta: 0 of 2 correct"))
        self.assertIn("For beta tasks,", first.system)
        self.assertIn("Correct answer: ok", parts["MISSED EXAMPLES OF THIS TASK TYPE"])
        self.assertNotIn("v-", first.prompt)   # validation and final tasks never appear
        self.assertNotIn("f-", first.prompt)
        second = next(r for r in model.requests if r.purpose == "revise_strategy:add_rule_2")
        self.assertTrue(sections(second.prompt)["TARGET TASK TYPE"].startswith("alpha:"))

    def test_rule_screens_and_truncation(self):
        cases = {
            "For beta tasks, set your forecasts to 30%.": "rule_off_lever",
            "=== DISCOVERY RESULTS THIS LEVEL ===": "rule_off_lever",
            "For beta tasks, compare with the reference answer.": "rule_off_lever",
            "For beta tasks, " + "x" * 400: "rule_too_long",
            "For beta tasks, retrain on more data.": "strategy_non_executable",
            "   \n": "rule_empty",
        }
        for reply, reason in cases.items():
            with self.subTest(reply=reply[:40]):
                result, _, _, _ = self.run_rules({"revise_strategy:add_rule": reply},
                                                 candidate_kinds=("add_rule",), max_depth=1)
                self.assertEqual(result.steps[0].candidates[0].reason, reason)
        INITIAL_WITH = replace(INITIAL, strategy="Solve each task carefully.")
        override = {"revise_strategy:add_rule": "Solve each task carefully."}
        result, _, _, _ = run_case(FakeModel(beta_needs_a_rule, lambda r: override.get(r.purpose)), two_type_plan(),
                                   C(candidate_kinds=("add_rule",), max_depth=1), initial=INITIAL_WITH)
        self.assertEqual(result.steps[0].candidates[0].reason, "unchanged")

        def truncated(text):
            def raise_it(req):
                if req.purpose == "revise_strategy:add_rule":
                    raise OutputTruncated("limit", text)
                return None
            return raise_it
        for partial, reason in (("For beta tasks, apply the beta-rule method.\nNow let me also", "accuracy_gain"),
                                ("For beta tasks, apply the beta-rule meth", "rule_truncated")):
            with self.subTest(partial=partial):
                result, _, _, _ = run_case(FakeModel(beta_needs_a_rule, truncated(partial)), two_type_plan(),
                                           C(candidate_kinds=("add_rule",), max_depth=1))
                self.assertEqual(result.steps[0].candidates[0].reason, reason)

    def test_recalibrate_keeps_only_task_type_lines_even_when_truncated(self):
        reply = ("alpha: about 9 in 10 correct (2 of 2 so far)\n- beta: about 2 in 10 correct (0 of 3 so far)\n"
                 "\n=== DISCOVERY RESULTS THIS LEVEL ===\nTask: Solve the distinct item d-b1.\nalpha: duplicate")

        def raise_it(req):
            if req.purpose == "revise_self_model:recalibrate":
                raise OutputTruncated("limit", reply)
            return None

        result, _, _, _ = run_case(FakeModel(beta_needs_a_rule, raise_it), two_type_plan(),
                                   C(candidate_kinds=("recalibrate",), max_depth=1))
        candidate = result.steps[0].candidates[0]
        self.assertTrue(candidate.eligible)
        self.assertEqual(candidate.self_model,
                         "alpha: about 9 in 10 correct (2 of 2 so far)\nbeta: about 2 in 10 correct (0 of 3 so far)")

    def test_whole_strategy_that_copies_the_evidence_is_ineligible(self):
        replies = {"revise_strategy:repair": "STRATEGY:\n=== DISCOVERY RESULTS THIS LEVEL ===\n[1] beta | INCORRECT"}
        result, _, _, _ = self.run_rules(replies, candidate_kinds=("repair",), max_depth=1)
        self.assertEqual(result.steps[0].candidates[0].reason, "strategy_echoes_context")


class IsolationAndScoringTests(unittest.TestCase):
    def test_resuming_rejects_task_ids_already_present_in_history(self):
        previous = RecursiveMirror(FakeModel(), make_plan(1), C(max_calls=4)).run(INITIAL)
        model = FakeModel()
        with self.assertRaises(ValueError):
            RecursiveMirror(model, make_plan(1), C(max_calls=4)).run(previous.committed_state)
        self.assertEqual(model.requests, [])

    def test_resuming_rejects_same_normalized_prompt_under_new_id(self):
        old_plan = make_plan(1)
        previous = RecursiveMirror(FakeModel(), old_plan, C(max_calls=4)).run(INITIAL)
        new_plan = Plan(
            (Round((Task("new-d", "New discovery item", "ok"),),
                   (Task("new-v", "New validation item", "ok"),)),),
            (Task("new-f", "  " + old_plan.final_tasks[0].prompt.upper() + "\n", "ok"),),
        )
        model = FakeModel()
        with self.assertRaises(ValueError):
            RecursiveMirror(model, new_plan, C(max_calls=4)).run(previous.committed_state)
        self.assertEqual(model.requests, [])

    def test_resuming_requires_complete_prompt_fingerprints(self):
        legacy = Observation(0, "discovery", "old-id", "general", "50", 0.5, "ok", True, "")
        state = replace(INITIAL, observations=(legacy,))
        model = FakeModel()
        with self.assertRaises(ValueError):
            RecursiveMirror(model, make_plan(1), C(max_calls=4)).run(state)
        self.assertEqual(model.requests, [])

    def test_resuming_with_fresh_tasks_preserves_prior_history(self):
        previous = RecursiveMirror(FakeModel(), make_plan(1), C(max_calls=4)).run(INITIAL)
        new_plan = Plan(
            (Round((Task("new-d", "New discovery item", "ok"),),
                   (Task("new-v", "New validation item", "ok"),)),),
            (Task("new-f", "New final item", "ok"),),
        )
        result = RecursiveMirror(FakeModel(), new_plan, C(max_calls=4)).run(previous.committed_state)
        self.assertEqual(len(result.committed_state.observations), 4)
        self.assertTrue(all(len(row.prompt_sha256) == 64 for row in result.committed_state.observations))
        self.assertEqual(result.committed_state.observations[:2], previous.committed_state.observations)

    def test_plan_rejects_duplicate_ids_and_normalized_prompts(self):
        d = Task("discovery", "Distinct discovery prompt", "a")
        v = Task("validation", "Distinct validation prompt", "b")
        for final in (Task("discovery", "Different final prompt", "c"),
                      Task("final", "  DISTINCT   validation\n prompt  ", "c")):
            with self.subTest(final=final):
                with self.assertRaisesRegex(ValueError, "disjoint"):
                    Plan((Round((d,), (v,)),), (final,))

    def test_every_split_must_be_nonempty(self):
        task = Task("only", "A prompt", "a")
        for rounds, final in (((), (task,)), ((Round((), (task,)),), (task,)),
                              ((Round((task,), ()),), (task,)),
                              ((Round((task,), (task,)),), ())):
            with self.subTest(rounds=rounds, final=final):
                with self.assertRaises(ValueError):
                    Plan(rounds, final)

    def test_answer_extraction_prefers_the_last_answer_line(self):
        cases = {
            "42": ("42", "last_line"),
            "  OK\n": ("OK", "last_line"),
            "Work: 6*7=42\nANSWER: 42": ("42", "answer_line"),
            "ANSWER: 41\nWait, recheck.\nANSWER: 42\n": ("42", "answer_line"),
            "**Final Answer:** 42.": ("42.", "answer_line"),
            "answer: \\boxed{29842}": ("29842", "answer_line"),
            "Reasoning\nANSWER:\n42": ("42", "last_line"),
            "   \n\n": ("", "empty"),
            "Step 3 done.\nANSWER: <41454>": ("41454", "answer_line"),
            "So the result is $340$.\n\\[ \\boxed{340} \\]": ("340", "boxed"),
            "Thus the final answer: Sunday": ("Sunday", "answer_line"),
            "ANSWER: $13.6$": ("13.6", "answer_line"),
            "ANSWER: \\(elbirroh\\)": ("elbirroh", "answer_line"),
            "ANSWER: \\text{Tuesday}": ("Tuesday", "answer_line"),
        }
        for response, expected in cases.items():
            with self.subTest(response=response):
                self.assertEqual(extract_answer(response), expected)

    def test_answer_normalization_is_deterministic_and_literal(self):
        same = [("29,842", "29842"), ("  Tenalp ", "tenalp"), ("**9.9**", "9.9"), ("'Sunday'.", "sunday"),
                ("`3`", "3"), ("New   York", "new york")]
        for raw, expected in same:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_answer(raw), expected)
        for left, right in [("nine", "9"), ("9.90", "9.9"), ("3,7", "37"), ("1,2345", "12345")]:
            with self.subTest(left=left):
                self.assertNotEqual(normalize_answer(left), normalize_answer(right))

    def test_forecast_parsing_accepts_only_one_integer_percentage(self):
        valid = {"85": 0.85, " 100\n": 1.0, "0": 0.0, "7%": 0.07, " 64 % ": 0.64, "**40%**": 0.4, "40%.": 0.4}
        for raw, probability in valid.items():
            with self.subTest(raw=raw):
                self.assertEqual(parse_forecast(raw), probability)
        for raw in ("216", "NaN", "0.5", "101", "-1", "050", "50 because likely", "", "85\n90",
                    "letter_count: 40%", "40% (1 of 2 correct so far)"):
            with self.subTest(raw=raw):
                self.assertIsNone(parse_forecast(raw))

    def test_invalid_forecasts_are_recorded_and_penalized(self):
        for raw in ("NaN", "0.5", "101", "-1", "050", "50 because likely"):
            with self.subTest(raw=raw):
                model = FakeModel(evaluation=lambda req, parts: raw if req.purpose == "forecast" else "ANSWER: ok")
                result = RecursiveMirror(model, make_plan(1), C(max_calls=4)).run(INITIAL)
                self.assertEqual(result.candidate_final_metrics.forecast_coverage, 0.0)
                self.assertEqual(result.candidate_final_metrics.brier, 1.0)
                self.assertFalse(result.transfer_improved)
                for row in result.committed_state.observations:
                    self.assertEqual(row.forecast_raw, raw)
                    self.assertIsNone(row.probability)

    def test_invalid_candidate_forecasts_block_selection_even_with_correct_answers(self):
        def evaluation(req, parts):
            if req.purpose == "answer":
                return "ANSWER: ok"
            return "50" if strategy_of(parts) == INITIAL.strategy else "certain"

        result = RecursiveMirror(FakeModel(evaluation), make_plan(1), C(candidate_kinds=("repair",))).run(INITIAL)
        self.assertFalse(result.steps[0].accepted)
        comparison = result.steps[0].candidates[0].comparison
        self.assertEqual((comparison.candidate.accuracy, comparison.candidate.forecast_coverage), (1.0, 0.0))
        # A strategy-only candidate is judged on answers; equal answers never pass, whatever the forecasts.
        self.assertEqual(comparison.reason, "no_declared_gain")
        self.assertEqual(result.committed_state.strategy, INITIAL.strategy)

    def test_paired_gate_requires_gain_and_disallows_regressions(self):
        cfg = C(min_accuracy_gain=0.1, min_brier_gain=0.05)
        parent = rows([(0.5, True), (0.5, False), (0.5, True), (0.5, False)])
        overconfident = rows([(0.0, True), (1.0, False), (0.0, True), (1.0, False)])
        cases = [
            (parent, rows([(0.5, True), (0.5, False), (0.5, True), (0.5, False)]), False, "no_declared_gain"),
            (parent, rows([(0.5, True), (0.5, True), (0.5, True), (0.5, False)]), True, "accuracy_gain"),
            (parent, rows([(0.8, True), (0.2, False), (0.8, True), (0.2, False)]), True, "calibration_gain"),
            (parent, rows([(0.5, True), (0.5, True), (0.9, False), (0.9, False)]), False, "calibration_regression"),
            (parent, rows([(0.0, False), (0.0, False), (0.0, False), (0.0, False)]), False, "accuracy_regression"),
            # One malformed forecast cannot veto a real accuracy gain ...
            (parent, rows([(1.0, True), (1.0, True), (None, True), (1.0, True)]), True, "accuracy_gain"),
            # ... accuracy is primary, so a real accuracy gain passes even with degraded forecasts,
            (parent, rows([(0.5, True), (0.5, True), (None, True), (None, False)]), True, "accuracy_gain"),
            # but with accuracy unchanged, malformed forecasts count as the worst loss,
            (parent, rows([(0.5, True), (0.5, False), (None, True), (None, False)]), False, "calibration_regression"),
            # a dropped forecast is allowed only when overall Brier (invalid = 1.0) also clearly falls,
            (overconfident, rows([(1.0, True), (0.0, False), (None, True), (0.0, False)]), True, "calibration_gain"),
            (parent, rows([(1.0, True), (0.0, False), (None, True), (0.0, False)]), False,
             "forecast_coverage_regression"),
            (overconfident, rows([(1.0, True), (0.0, False), (1.0, True), (0.0, False)]), True, "calibration_gain"),
        ]
        for base, candidate, passed, reason in cases:
            with self.subTest(reason=reason, candidate=candidate):
                result = compare(base, candidate, cfg)
                self.assertEqual((result.passed, result.reason), (passed, reason))
        with self.assertRaisesRegex(ValueError, "same tasks"):
            compare(parent, parent[::-1], cfg)

    def test_gate_judges_each_changed_lever_where_it_acts(self):
        cfg = Config()
        initial = rows([(0.9, True)] * 8 + [(0.9, False)] * 8)
        # Same answers, honest forecasts: a self-model change passes on calibration.
        recalibrated = rows([(0.55, True)] * 8 + [(0.55, False)] * 8)
        self.assertEqual(compare(initial, recalibrated, cfg, frozenset({"self_model"})).reason, "calibration_gain")
        # A composite change with one extra (non-significant) correct answer must not lose that
        # calibration gain: the gate is monotonic in accuracy (multi-agent review, high).
        composite = rows([(0.55, True)] * 9 + [(0.55, False)] * 7)
        result = compare(initial, composite, cfg, frozenset({"strategy", "self_model"}))
        self.assertEqual((result.passed, result.reason), (True, "calibration_gain"))
        # A strategy-only change is judged on answers: reshuffled answers with better-looking
        # forecasts are not an improvement.
        reshuffled = rows([(0.6, True)] * 6 + [(0.6, False)] * 2 + [(0.6, True)] * 2 + [(0.6, False)] * 6)
        self.assertEqual(compare(initial, reshuffled, cfg, frozenset({"strategy"})).reason, "no_declared_gain")
        # One malformed forecast out of 24 does not veto a clear calibration gain.
        base = rows([(0.9, True)] * 12 + [(0.9, False)] * 12)
        echoed = rows([(0.5, True)] * 12 + [(0.5, False)] * 11 + [(None, False)])
        self.assertEqual(compare(base, echoed, cfg, frozenset({"self_model"})).reason, "calibration_gain")
        # Identical policies cannot pass.
        self.assertEqual(compare(base, base, cfg, frozenset()).reason, "no_declared_gain")

    def test_evidence_gate_needs_consistent_per_task_gains(self):
        cfg = Config()  # max_p_value = 0.25
        parent = rows([(0.5, False)] * 4 + [(0.5, True)] * 4)
        lucky = rows([(0.5, True)] + [(0.5, False)] * 3 + [(0.5, True)] * 4)          # one net win
        lucky_mixed = rows([(0.5, True)] * 3 + [(0.5, False)] + [(0.5, False)] + [(0.5, True)] * 3)
        consistent = rows([(0.5, True)] * 3 + [(0.5, False)] + [(0.5, True)] * 4)     # three wins
        for candidate, passed, reason, p in [
            (lucky, False, "gain_not_consistent", 0.5),
            (lucky_mixed, False, "gain_not_consistent", 5 / 16),
            (consistent, True, "accuracy_gain", 0.125),
        ]:
            with self.subTest(reason=reason, p=p):
                result = compare(parent, candidate, cfg)
                self.assertEqual((result.passed, result.reason), (passed, reason))
                self.assertAlmostEqual(result.accuracy_p, p)
        # A uniform shift of forecasts toward the observed rate is a consistent calibration gain.
        miscalibrated = rows([(0.9, False)] * 6 + [(0.9, True)] * 2)
        recalibrated = rows([(0.3, False)] * 6 + [(0.3, True)] * 2)
        result = compare(miscalibrated, recalibrated, cfg)
        self.assertEqual((result.passed, result.reason, result.paired_n), (True, "calibration_gain", 8))
        self.assertLessEqual(result.calibration_p, 0.25)
        one_item = compare(miscalibrated[:1], recalibrated[:1], cfg)
        self.assertEqual((one_item.passed, one_item.reason, one_item.calibration_p),
                         (False, "gain_not_consistent", 0.5))

    def test_invalid_forecasts_are_not_excluded_from_average(self):
        valid = Observation(0, "discovery", "a", "general", "100", 1.0, "ok", True, "0" * 64)
        invalid = replace(valid, task_id="b", forecast_raw="uncertain", probability=None)
        self.assertEqual(summarize((valid, invalid)), Metrics(1.0, 0.5, 0.5, 2, 1.0))

    def test_paired_p_value_is_exact_one_sided_and_deterministic(self):
        self.assertEqual(paired_p_value([]), 1.0)
        self.assertEqual(paired_p_value([0, 0.0]), 1.0)
        self.assertEqual(paired_p_value([1, 1, 1]), 0.125)              # exact sign test
        self.assertAlmostEqual(paired_p_value([1] * 5 + [-1]), 7 / 64)
        self.assertEqual(paired_p_value([-1] * 4), 1.0)
        self.assertEqual(paired_p_value([1] * 30 + [-1] * 10), paired_p_value([-1] * 10 + [1] * 30))
        self.assertAlmostEqual(paired_p_value([0.6, 0.6, -0.1]), 2 / 8)   # magnitude matters
        many = [0.3, -0.05, 0.2, 0.1] * 6
        self.assertEqual(paired_p_value(many), paired_p_value(many))     # fixed-seed Monte Carlo
        self.assertLess(paired_p_value(many), 0.01)

    def test_config_rejects_unknown_or_duplicate_candidate_kinds(self):
        for kinds in ((), ("repair", "repair"), ("mutate",), ["repair"]):
            with self.subTest(kinds=kinds), self.assertRaises(ValueError):
                Config(candidate_kinds=kinds)
        with self.assertRaises(ValueError):
            Config(reveal_discovery_answers=1)


if __name__ == "__main__":
    unittest.main()
