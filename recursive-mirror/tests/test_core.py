"""Independent behavioural tests; no model weights or network are required."""
from __future__ import annotations

import inspect
import json
import unittest
from dataclasses import replace

from recursive_mirror.core import (
    Config, Metrics, Observation, Plan, RecursiveMirror, Round, State, Task,
    digest, improves, summarize,
)


def make_plan(rounds=3, final_count=1, expected="ok"):
    def task(name):
        return Task(name, f"Solve the distinct item {name}.", expected)

    return Plan(
        tuple(Round((task(f"discovery-{i}"),), (task(f"validation-{i}"),))
              for i in range(rounds)),
        tuple(task(f"final-{i}") for i in range(final_count)),
    )


class FakeModel:
    """Returns deterministic text and records only the public request boundary."""

    identity = "deterministic-test-model"
    views = {
        "actor": "From the actor angle, this is one reasoning agent.",
        "reviewer": "From the reviewer angle, this is two distinct reasoning roles.",
        "whole_system": "Both descriptions can apply when their viewpoints differ.",
    }

    def __init__(self, evaluation=None, override=None):
        self.requests = []
        self.live_recursion_depths = []
        self.evaluation = evaluation or self.improving_evaluation
        self.override = override

    @staticmethod
    def improving_evaluation(request, data):
        version = int(data["agent"]["strategy"].removeprefix("strategy-"))
        if request.purpose == "answer":
            return "ok"
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
        if request.purpose in ("forecast", "answer"):
            return self.evaluation(request, json.loads(request.prompt))
        if request.purpose.startswith("perspective:"):
            return self.views[request.purpose.split(":", 1)[1]]
        if request.purpose == "revise_strategy":
            return f"strategy-{json.loads(request.prompt)['frame']['round_index'] + 1}"
        if request.purpose == "revise_self_model":
            return f"self-model-{json.loads(request.prompt)['frame']['round_index'] + 1}"
        raise AssertionError(f"Unexpected model purpose: {request.purpose}")


INITIAL = State("strategy-0", "self-model-0")


class RecursiveExecutionTests(unittest.TestCase):
    def run_case(self, model=None, plan=None, config=None):
        model = model or FakeModel()
        events = []
        engine = RecursiveMirror(model, plan or make_plan(), config, events.append)
        return engine.run(INITIAL), model, events, engine

    def test_calls_are_genuinely_nested_and_updated_state_reaches_next_level(self):
        result, model, events, _ = self.run_case()
        discovery = [
            (json.loads(req.prompt)["agent"]["strategy"], depth)
            for req, depth in zip(model.requests, model.live_recursion_depths)
            if req.purpose == "forecast"
            and json.loads(req.prompt)["task"]["task_id"].startswith("discovery-")
        ]
        self.assertEqual(discovery, [
            ("strategy-0", 1), ("strategy-1", 2), ("strategy-2", 3),
        ])
        nesting = [(e["event"], e["round_index"]) for e in events
                   if e["event"] in ("recursive_enter", "recursive_return")]
        self.assertEqual(nesting, [
            ("recursive_enter", 0), ("recursive_enter", 1),
            ("recursive_enter", 2), ("recursive_return", 2),
            ("recursive_return", 1), ("recursive_return", 0),
        ])
        self.assertEqual(result.status, "plan_exhausted")
        self.assertEqual(result.calls, 37)
        self.assertTrue(all(step.accepted for step in result.steps))
        self.assertTrue(result.transfer_improved)
        self.assertEqual(result.committed_state.strategy, "strategy-3")
        self.assertEqual(result.committed_state.version, 3)
        self.assertEqual(result.claim, "measured_agent_strategy_and_self_prediction_only")

    def test_depth_limit_bounds_nested_calls(self):
        result, model, _, _ = self.run_case(
            plan=make_plan(4), config=Config(max_depth=2),
        )
        self.assertEqual(result.status, "depth_limit")
        self.assertEqual(len(result.steps), 2)
        self.assertEqual(max(model.live_recursion_depths), 2)

    def test_plateau_stops_after_declared_number_of_rejections(self):
        model = FakeModel(evaluation=lambda req, data: "ok" if req.purpose == "answer" else "50")
        result, _, _, _ = self.run_case(
            model, make_plan(5), Config(max_depth=5, patience=2),
        )
        self.assertEqual(result.status, "no_measured_progress")
        self.assertEqual(len(result.steps), 2)
        self.assertFalse(any(step.accepted for step in result.steps))
        self.assertFalse(result.transfer_improved)
        self.assertEqual(result.committed_state.strategy, INITIAL.strategy)

    def test_proposal_is_frozen_before_any_validation_request(self):
        result, model, events, _ = self.run_case(plan=make_plan(1))
        proposal_pos = next(i for i, e in enumerate(events) if e["event"] == "proposal")
        validation_positions = [
            i for i, e in enumerate(events)
            if e["event"] == "llm_request"
            and e["request"]["purpose"] in ("forecast", "answer")
            and json.loads(e["request"]["prompt"])["task"]["task_id"] == "validation-0"
        ]
        self.assertEqual(len(validation_positions), 4)
        self.assertTrue(all(i > proposal_pos for i in validation_positions))
        proposal = events[proposal_pos]
        self.assertEqual(proposal["state_sha256"], digest(proposal["state"]))
        self.assertEqual(result.steps[0].proposed_state_sha256, proposal["state_sha256"])
        for request in model.requests:
            if request.purpose.startswith("perspective:") or request.purpose.startswith("revise_"):
                self.assertNotIn("validation-0", request.prompt)
                self.assertNotIn("final-0", request.prompt)

    def test_each_reflection_receives_its_discovery_questions_and_measured_answers_only(self):
        secret = "PRIVATE_ANSWER_KEY_594891"
        basic = make_plan(2, expected=secret)
        plan = replace(basic, rounds=tuple(
            replace(round_, discovery=round_.discovery + (
                Task(f"extra-discovery-{index}", f"Additional discovery question {index}?", secret),
            ))
            for index, round_ in enumerate(basic.rounds)
        ))
        model = FakeModel(evaluation=lambda req, data: "50" if req.purpose == "forecast" else "observed guess")
        _, _, events, _ = self.run_case(model, plan)
        reflection_requests = [req for req in model.requests
                               if req.purpose.startswith("perspective:")
                               or req.purpose.startswith("revise_")]
        self.assertEqual(len(reflection_requests), 10)
        hidden_tasks = [task for round_ in plan.rounds for task in round_.validation]
        hidden_tasks.extend(plan.final_tasks)
        for request in reflection_requests:
            evidence = json.loads(request.prompt)
            index = evidence["frame"]["round_index"]
            discovery = plan.rounds[index].discovery
            self.assertEqual(evidence["discovery_tasks"], [
                {"task_id": task.task_id, "prompt": task.prompt, "category": task.category}
                for task in discovery
            ])
            measured = [{key: value for key, value in event.items() if key != "event"}
                        for event in events if event["event"] == "observation"
                        and event["phase"] == "discovery" and event["round_index"] == index]
            self.assertEqual(evidence["measured_observations"], measured)
            self.assertEqual([row["task_id"] for row in measured], [task.task_id for task in discovery])
            self.assertTrue(all(row["answer"] == "observed guess" and not row["correct"]
                                for row in measured))
            self.assertNotIn(secret, request.prompt)
            self.assertNotIn('"expected"', request.prompt)
            for task in hidden_tasks:
                self.assertNotIn(task.prompt, request.prompt)
                self.assertNotIn(task.task_id, request.prompt)

    def test_same_validation_item_uses_paired_seeds(self):
        _, model, _, _ = self.run_case(plan=make_plan(1))
        for purpose in ("forecast", "answer"):
            requests = [req for req in model.requests if req.purpose == purpose
                        and json.loads(req.prompt)["task"]["task_id"] == "validation-0"]
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[0].seed, requests[1].seed)
            self.assertEqual(json.loads(requests[0].prompt)["task"],
                             json.loads(requests[1].prompt)["task"])
            self.assertNotEqual(json.loads(requests[0].prompt)["agent"],
                                json.loads(requests[1].prompt)["agent"])

    def test_final_heldout_failure_rolls_back_only_the_policy(self):
        def evaluation(req, data):
            candidate = data["agent"]["strategy"] != INITIAL.strategy
            final = data["task"]["task_id"].startswith("final-")
            if req.purpose == "forecast":
                return "100"
            return "ok" if candidate != final else "incorrect"

        result, _, events, _ = self.run_case(FakeModel(evaluation), make_plan(1))
        self.assertTrue(result.steps[0].accepted)
        self.assertFalse(result.transfer_improved)
        self.assertEqual(result.candidate_state.strategy, "strategy-1")
        self.assertEqual(result.committed_state.strategy, INITIAL.strategy)
        self.assertEqual(result.committed_state.self_model, INITIAL.self_model)
        self.assertEqual(result.committed_state.version, INITIAL.version)
        self.assertEqual(len(result.committed_state.perspectives), 3)
        self.assertEqual(len(result.committed_state.observations), 5)
        self.assertEqual(result.initial_final_metrics.accuracy, 1.0)
        self.assertEqual(result.candidate_final_metrics.accuracy, 0.0)
        self.assertEqual(events[-1]["event"], "run_completed")

    def test_rejected_revision_retains_all_opposing_perspectives(self):
        model = FakeModel(evaluation=lambda req, data: "ok" if req.purpose == "answer" else "50")
        result, _, _, _ = self.run_case(model, make_plan(2))
        self.assertTrue(all(not step.accepted for step in result.steps))
        self.assertEqual(result.candidate_state.version, 0)
        self.assertEqual(result.candidate_state.strategy, INITIAL.strategy)
        self.assertEqual(len(result.committed_state.perspectives), 6)
        self.assertEqual([p.text for p in result.committed_state.perspectives],
                         list(FakeModel.views.values()) * 2)
        self.assertEqual([p.round_index for p in result.committed_state.perspectives],
                         [0, 0, 0, 1, 1, 1])
        self.assertEqual(len(result.committed_state.observations), 8)

    def test_later_reflection_receives_previous_angles_without_unbounded_repetition(self):
        _, model, _, _ = self.run_case(plan=make_plan(3))
        reflections = [req for req in model.requests if req.purpose == "perspective:actor"]
        self.assertEqual(len(reflections), 3)
        for text in FakeModel.views.values():
            self.assertNotIn(text, reflections[0].prompt)
            self.assertEqual(reflections[1].prompt.count(text), 1)
            self.assertEqual(reflections[2].prompt.count(text), 1)

    def test_instance_is_one_use_after_success(self):
        _, model, events, engine = self.run_case(plan=make_plan(1))
        calls, event_count = len(model.requests), len(events)
        with self.assertRaisesRegex(RuntimeError, "only once"):
            engine.run(INITIAL)
        self.assertEqual(len(model.requests), calls)
        self.assertEqual(len(events), event_count)


class BudgetAndAuditTests(unittest.TestCase):
    def test_budget_must_cover_final_comparison(self):
        with self.assertRaisesRegex(ValueError, "reserved final"):
            RecursiveMirror(FakeModel(), make_plan(final_count=2), Config(max_calls=7))

    def test_final_comparison_is_preserved_when_no_round_fits(self):
        model, events = FakeModel(), []
        result = RecursiveMirror(model, make_plan(), Config(max_calls=4), events.append).run(INITIAL)
        self.assertEqual(result.status, "call_budget")
        self.assertEqual(result.calls, 4)
        self.assertEqual(result.steps, ())
        self.assertEqual([req.purpose for req in model.requests],
                         ["forecast", "answer", "forecast", "answer"])
        self.assertFalse(any(e["event"] == "recursive_enter" for e in events))

    def test_round_cannot_consume_final_reserve(self):
        model = FakeModel()
        result = RecursiveMirror(model, make_plan(3), Config(max_calls=25)).run(INITIAL)
        self.assertEqual(result.status, "call_budget")
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.calls, 15)
        self.assertEqual([json.loads(req.prompt)["task"]["task_id"]
                          for req in model.requests[-4:]], ["final-0"] * 4)

    def test_exact_budget_allows_complete_round_and_final_test(self):
        result = RecursiveMirror(FakeModel(), make_plan(1), Config(max_calls=15)).run(INITIAL)
        self.assertEqual(result.calls, 15)
        self.assertEqual(result.status, "plan_exhausted")
        self.assertTrue(result.transfer_improved)

    def test_invalid_raw_output_is_audited_before_failure(self):
        for bad_response in ("", " " * 4, "x" * 12001):
            with self.subTest(length=len(bad_response)):
                model = FakeModel(override=lambda req: bad_response)
                events = []
                engine = RecursiveMirror(model, make_plan(1), emit=events.append)
                with self.assertRaises(ValueError):
                    engine.run(INITIAL)
                responses = [e for e in events if e["event"] == "llm_response"]
                self.assertEqual(responses[0]["response"], bad_response)
                self.assertEqual(events[-1]["event"], "run_failed")
                self.assertEqual(events[-1]["calls"], 1)
                self.assertFalse(any(e["event"] == "run_completed" for e in events))
                with self.assertRaisesRegex(RuntimeError, "only once"):
                    engine.run(INITIAL)
                self.assertEqual(len(model.requests), 1)

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

    def test_backend_receipt_is_saved_alongside_each_raw_response(self):
        model = FakeModel()
        receipt = {"input_tokens": 42, "output_tokens": 1, "finish_reason": "eos"}
        model.last_receipt = receipt
        events = []
        RecursiveMirror(model, make_plan(1), Config(max_calls=4), events.append).run(INITIAL)
        responses = [event for event in events if event["event"] == "llm_response"]
        self.assertEqual(len(responses), 4)
        self.assertTrue(all(event["backend_receipt"] == receipt for event in responses))
        self.assertEqual([event["response"] for event in responses], ["10", "ok", "10", "ok"])

    def test_oversized_request_stops_before_backend_without_truncation(self):
        model, events = FakeModel(), []
        engine = RecursiveMirror(model, make_plan(1), Config(max_request_chars=100), events.append)
        with self.assertRaisesRegex(ValueError, "no truncation"):
            engine.run(INITIAL)
        self.assertEqual(model.requests, [])
        self.assertEqual(events[-1]["event"], "run_failed")
        self.assertEqual(events[-1]["calls"], 0)


class IsolationAndScoringTests(unittest.TestCase):
    def test_resuming_rejects_task_ids_already_present_in_history(self):
        previous = RecursiveMirror(FakeModel(), make_plan(1), Config(max_calls=4)).run(INITIAL)
        model = FakeModel()
        with self.assertRaises(ValueError):
            RecursiveMirror(model, make_plan(1), Config(max_calls=4)).run(previous.committed_state)
        self.assertEqual(model.requests, [])

    def test_resuming_rejects_same_normalized_prompt_under_new_id(self):
        old_plan = make_plan(1)
        previous = RecursiveMirror(FakeModel(), old_plan, Config(max_calls=4)).run(INITIAL)
        new_plan = Plan(
            (Round((Task("new-d", "New discovery item", "ok"),),
                   (Task("new-v", "New validation item", "ok"),)),),
            (Task("new-f", "  " + old_plan.final_tasks[0].prompt.upper() + "\n", "ok"),),
        )
        model = FakeModel()
        with self.assertRaises(ValueError):
            RecursiveMirror(model, new_plan, Config(max_calls=4)).run(previous.committed_state)
        self.assertEqual(model.requests, [])

    def test_resuming_requires_complete_prompt_fingerprints(self):
        legacy = Observation(0, "discovery", "old-id", "general", "50", 0.5, "ok", True)
        state = replace(INITIAL, observations=(legacy,))
        model = FakeModel()
        with self.assertRaises(ValueError):
            RecursiveMirror(model, make_plan(1), Config(max_calls=4)).run(state)
        self.assertEqual(model.requests, [])

    def test_resuming_with_fresh_tasks_preserves_prior_history(self):
        previous = RecursiveMirror(FakeModel(), make_plan(1), Config(max_calls=4)).run(INITIAL)
        new_plan = Plan(
            (Round((Task("new-d", "New discovery item", "ok"),),
                   (Task("new-v", "New validation item", "ok"),)),),
            (Task("new-f", "New final item", "ok"),),
        )
        result = RecursiveMirror(FakeModel(), new_plan, Config(max_calls=4)).run(previous.committed_state)
        self.assertEqual(len(result.committed_state.observations), 4)
        self.assertTrue(all(len(row.prompt_sha256) == 64 for row in result.committed_state.observations))
        self.assertEqual(result.committed_state.observations[:2], previous.committed_state.observations)

    def test_plan_rejects_duplicate_ids_and_normalized_prompts(self):
        d = Task("discovery", "Distinct discovery prompt", "a")
        v = Task("validation", "Distinct validation prompt", "b")
        for final in (
            Task("discovery", "Different final prompt", "c"),
            Task("final", "  DISTINCT   validation\n prompt  ", "c"),
        ):
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

    def test_evaluator_answer_keys_never_enter_requests(self):
        expected = "HIDDEN_EVALUATOR_SECRET_726403"
        model = FakeModel(evaluation=lambda req, data: "50" if req.purpose == "forecast" else "guess")
        RecursiveMirror(model, make_plan(2, expected=expected)).run(INITIAL)
        for req in model.requests:
            self.assertNotIn(expected, req.prompt)
            self.assertNotIn(expected, req.system)
            self.assertNotIn('"expected"', req.prompt)
            if req.purpose in ("forecast", "answer"):
                self.assertEqual(set(json.loads(req.prompt)["task"]),
                                 {"task_id", "prompt", "category"})

    def test_noninteger_forecasts_are_recorded_and_penalized(self):
        for raw in ("NaN", "0.5", "101", "-1", "50%", "050", "50 because likely"):
            with self.subTest(raw=raw):
                model = FakeModel(evaluation=lambda req, data: raw if req.purpose == "forecast" else "ok")
                result = RecursiveMirror(model, make_plan(1), Config(max_calls=4)).run(INITIAL)
                self.assertEqual(result.candidate_final_metrics.forecast_coverage, 0.0)
                self.assertEqual(result.candidate_final_metrics.brier, 1.0)
                self.assertFalse(result.transfer_improved)
                for row in result.committed_state.observations:
                    self.assertEqual(row.forecast_raw, raw)
                    self.assertIsNone(row.probability)

    def test_invalid_candidate_forecast_blocks_selection_even_with_correct_answers(self):
        def evaluation(req, data):
            if req.purpose == "answer":
                return "ok"
            return "50" if data["agent"]["strategy"] == INITIAL.strategy else "certain"

        result = RecursiveMirror(FakeModel(evaluation), make_plan(1)).run(INITIAL)
        self.assertFalse(result.steps[0].accepted)
        self.assertEqual(result.steps[0].candidate_metrics.accuracy, 1.0)
        self.assertEqual(result.steps[0].candidate_metrics.forecast_coverage, 0.0)
        self.assertEqual(result.committed_state.strategy, INITIAL.strategy)

    def test_trimmed_integer_forecasts_and_normalized_answers_are_accepted(self):
        model = FakeModel(evaluation=lambda req, data: " 100\n" if req.purpose == "forecast" else "  OK\n")
        result = RecursiveMirror(model, make_plan(1), Config(max_calls=4)).run(INITIAL)
        self.assertEqual(result.candidate_final_metrics, Metrics(1.0, 0.0, 1.0, 1))

    def test_metric_rule_requires_gain_and_disallows_regressions(self):
        parent = Metrics(0.5, 0.25, 1.0, 10)
        cfg = Config(min_accuracy_gain=0.1, min_brier_gain=0.1)
        cases = [
            (Metrics(0.5, 0.25, 1.0, 10), False),
            (Metrics(0.6, 0.25, 1.0, 10), True),
            (Metrics(0.5, 0.15, 1.0, 10), True),
            (Metrics(0.6, 0.26, 1.0, 10), False),
            (Metrics(0.4, 0.0, 1.0, 10), False),
            (Metrics(1.0, 0.0, 0.9, 10), False),
        ]
        for candidate, expected in cases:
            with self.subTest(candidate=candidate):
                self.assertEqual(improves(parent, candidate, cfg), expected)

    def test_invalid_forecasts_are_not_excluded_from_average(self):
        valid = Observation(0, "test", "a", "general", "100", 1.0, "ok", True)
        invalid = replace(valid, task_id="b", forecast_raw="uncertain", probability=None)
        self.assertEqual(summarize((valid, invalid)), Metrics(1.0, 0.5, 0.5, 2))


if __name__ == "__main__":
    unittest.main()
