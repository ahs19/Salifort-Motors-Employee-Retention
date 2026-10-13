from __future__ import annotations

from contextlib import nullcontext, redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from recursive_mirror.cli import main
from recursive_mirror.core import (Attempt, Config, OutputTruncated, Perspective, RecursiveMirror, Request,
                                   State, canonical, digest, fingerprint)
from recursive_mirror.demo import ScriptedTestDouble, demo_plan
from recursive_mirror.runtime import (GenerationFailure, HuggingFaceBackend, checkpoint_for,
                                     execute, load_checkpoint, load_plan, plan_from_dict, read_json,
                                     render_report)


ROOT = Path(__file__).resolve().parents[1]
V08_CHECKPOINT = ROOT / "evidence" / "v0.8.0-colab-20261009" / "token768_followup" / "trial" / "checkpoint.json"


def demo_result():
    return RecursiveMirror(ScriptedTestDouble(), demo_plan(), Config(max_depth=2, max_calls=200)).run()


def legacy_checkpoint(state_dict):
    """A checkpoint in the exact v0.8 (format v1) layout."""
    payload = {"format": "recursive-mirror-checkpoint-v1", "model_identity": "legacy-model",
               "plan_sha256": "b" * 64, "state": state_dict, "state_sha256": digest(state_dict)}
    return {**payload, "checkpoint_sha256": digest(payload)}


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_json(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_demo_persists_recursive_events_report_and_committed_state(self):
        output = self.root / "demo"
        result = execute(ScriptedTestDouble(), demo_plan(), output, Config(max_depth=2, max_calls=200))
        self.assertTrue(result.transfer_improved)
        # Level 0 repairs the strategy; level 1 recalibrates the self-model to the new strategy.
        self.assertEqual([s.accepted for s in result.steps], [True, True])
        self.assertEqual([s.selected_kind for s in result.steps], ["repair", "recalibrate"])
        self.assertEqual([[c.reason for c in s.candidates] for s in result.steps],
                         [["accuracy_gain", "calibration_gain"], ["unchanged", "calibration_gain"]])
        self.assertGreater(result.cache_hits, 0)
        self.assertEqual(result.model_identity, "scripted-test-double")
        self.assertEqual(load_checkpoint(output / "checkpoint.json"), result.committed_state)
        events = [json.loads(line) for line in (output / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(sum(e["event"] == "recursive_enter" for e in events), 2)
        self.assertEqual(sum(e["event"] == "llm_response" for e in events), result.calls)
        self.assertEqual(sum(e["event"] == "llm_cache_hit" for e in events), result.cache_hits)
        self.assertEqual(load_plan(output / "plan.json"), demo_plan())
        self.assertEqual(events[-1]["event"], "runtime_completed")
        report = (output / "report.md").read_text(encoding="utf-8")
        self.assertIn("answered more held-out final tasks correctly (0% -> 100%) and was committed", report)
        self.assertIn("## Initial policy", report)
        self.assertIn(State().strategy, report)
        self.assertIn("| repair | **selected** (`accuracy_gain`) | 0% -> 100%", report)
        self.assertIn(ScriptedTestDouble.FIXED, report)
        self.assertIn("Nothing here certifies awareness", report)

    def test_existing_output_is_not_overwritten(self):
        output = self.root / "exists"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("untouched", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            execute(ScriptedTestDouble(), demo_plan(), output)
        self.assertEqual(marker.read_text(encoding="utf-8"), "untouched")
        self.assertEqual(list(output.iterdir()), [marker])

    def test_checkpoint_rejects_tamper_and_type_confusion(self):
        original = checkpoint_for(demo_result())
        self.assertEqual(original["format"], "recursive-mirror-checkpoint-v2")
        mutations = [
            lambda obj: obj["state"].update(version=True),
            lambda obj: obj["state"]["observations"][0].update(correct="false"),
            lambda obj: obj["state"]["observations"][0].update(probability=True),
            lambda obj: obj["state"]["observations"][0].update(probability=0.1),
            lambda obj: obj["state"]["observations"][0].update(arm="repair"),
            lambda obj: obj["state"]["observations"][0].update(answer_status="guessed"),
            lambda obj: obj["state"]["observations"][0].update(answer="", correct=True),
            lambda obj: obj["state"]["perspectives"][0].update(round_index=False),
            lambda obj: obj["state"]["perspectives"][0].update(angle=[]),
            lambda obj: obj["state"]["perspectives"][0].update(text="", complete=True),
            lambda obj: obj["state"]["attempts"][0].update(outcome="promoted"),
            lambda obj: obj["state"]["attempts"][0].update(kind="final_transfer"),
            lambda obj: obj["state"]["attempts"][0].update(candidate_accuracy=1.5),
            lambda obj: obj["state"]["attempts"][0].update(extra="injected"),
            lambda obj: obj["state"].update(extra="injected"),
            lambda obj: obj["state"].pop("attempts"),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                obj = json.loads(json.dumps(original))
                mutate(obj)
                obj["state_sha256"] = digest(obj["state"])
                obj["checkpoint_sha256"] = digest({k: v for k, v in obj.items() if k != "checkpoint_sha256"})
                with self.assertRaises(ValueError):
                    load_checkpoint(self.write_json("mutated.json", obj))
        original["state"]["strategy"] = "Tampered strategy"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_checkpoint(self.write_json("tampered.json", original))

    def test_checkpoint_history_limits_match_on_save_and_load(self):
        result = demo_result()
        view = Perspective(0, "actor", "a" * 64, "Retained perspective")
        attempt = Attempt(0, "repair", "s", "m", "rejected", "no_declared_gain")
        for field, item in (("perspectives", view), ("observations", result.committed_state.observations[0]),
                            ("attempts", attempt)):
            with self.subTest(field=field):
                state = State(perspectives=(), observations=(), attempts=())
                at_limit = replace(result, committed_state=replace(state, **{field: (item,) * 2}))
                beyond = replace(result, committed_state=replace(state, **{field: (item,) * 3}))
                # Exercise the same shared cap at a small bound, preserving all
                # entries. The production constant is 10,000 for each history.
                with patch("recursive_mirror.runtime.MAX_COLLECTION_ITEMS", 2):
                    checkpoint = checkpoint_for(at_limit)
                    self.assertEqual(load_checkpoint(self.write_json("at-limit.json", checkpoint)), at_limit.committed_state)
                    with self.assertRaisesRegex(ValueError, "at most 2 items"):
                        checkpoint_for(beyond)
                unchecked = checkpoint_for(beyond)
                with patch("recursive_mirror.runtime.MAX_COLLECTION_ITEMS", 2):
                    with self.assertRaisesRegex(ValueError, "at most 2 items"):
                        load_checkpoint(self.write_json("too-many.json", unchecked))

    def test_checkpoint_utf8_byte_limit_includes_final_newline(self):
        result = replace(demo_result(), committed_state=State(strategy="Unicode strategy: π"))
        checkpoint = checkpoint_for(result)
        encoded = (canonical(checkpoint) + "\n").encode("utf-8")
        path = self.root / "byte-limit.json"
        path.write_bytes(encoded)
        with patch("recursive_mirror.runtime.MAX_JSON_BYTES", len(encoded)):
            self.assertEqual(checkpoint_for(result), checkpoint)
            self.assertEqual(load_checkpoint(path), result.committed_state)
        with patch("recursive_mirror.runtime.MAX_JSON_BYTES", len(encoded) - 1):
            with self.assertRaisesRegex(ValueError, "reload limit"):
                checkpoint_for(result)
            with self.assertRaisesRegex(ValueError, "size limit"):
                load_checkpoint(path)

    def test_v1_checkpoints_upgrade_with_their_original_hashes(self):
        state = {"strategy": "Old strategy", "self_model": "Old self-model", "version": 2,
                 "perspectives": [{"round_index": 0, "angle": "actor", "state_sha256": "c" * 64, "text": "view"}],
                 "observations": [{"round_index": 0, "phase": "discovery", "task_id": "old", "category": "a",
                                   "forecast_raw": "216", "probability": None, "answer": "43",
                                   "correct": True, "prompt_sha256": "d" * 64}]}
        loaded = load_checkpoint(self.write_json("v1.json", legacy_checkpoint(state)))
        self.assertEqual((loaded.strategy, loaded.version, loaded.attempts), ("Old strategy", 2, ()))
        self.assertTrue(loaded.perspectives[0].complete)
        self.assertEqual((loaded.observations[0].answer_status, loaded.observations[0].policy_sha256),
                         ("legacy", ""))
        tampered = legacy_checkpoint(state)
        tampered["state"]["version"] = 3
        tampered["checkpoint_sha256"] = digest({k: v for k, v in tampered.items() if k != "checkpoint_sha256"})
        with self.assertRaisesRegex(ValueError, "state hash mismatch"):
            load_checkpoint(self.write_json("v1-tampered.json", tampered))

    @unittest.skipUnless(V08_CHECKPOINT.is_file(), "v0.8 Colab evidence not shipped with this archive")
    def test_real_v08_colab_checkpoint_loads_and_can_be_continued(self):
        state = load_checkpoint(V08_CHECKPOINT)
        self.assertEqual(state.strategy, State().strategy)
        self.assertEqual(len(state.perspectives), 6)
        self.assertEqual(len(state.observations), 2 + 4 + 4)
        self.assertEqual(state.attempts, ())
        result = RecursiveMirror(ScriptedTestDouble(), demo_plan(), Config(max_depth=1, max_calls=200)).run(state)
        self.assertEqual(result.committed_state.observations[:10], state.observations)

    def test_checkpoint_continuation_requires_same_model_identity(self):
        result = demo_result()
        path = self.write_json("checkpoint.json", checkpoint_for(result))
        self.assertEqual(load_checkpoint(path, expected_model_identity="scripted-test-double"), result.committed_state)
        with self.assertRaisesRegex(ValueError, "model identity"):
            load_checkpoint(path, expected_model_identity="different-model-or-revision")
        self.assertEqual(load_checkpoint(path), result.committed_state)

    def test_cli_rejects_cross_model_state_before_loading_or_creating_output(self):
        checkpoint = self.write_json("checkpoint.json", checkpoint_for(demo_result()))
        plan = self.write_json("plan.json", asdict(demo_plan()))
        output = self.root / "cross-model"
        backend = Mock(identity="different-model-or-revision")
        with patch("recursive_mirror.cli.HuggingFaceBackend", return_value=backend):
            with redirect_stderr(io.StringIO()) as stderr:
                code = main(["run", "--plan", str(plan), "--output", str(output),
                             "--model", "different/model", "--state", str(checkpoint)])
        self.assertEqual(code, 1)
        self.assertIn("model identity", stderr.getvalue())
        self.assertFalse(output.exists())
        backend.complete.assert_not_called()

    def test_unreloadable_result_is_not_published_as_completed(self):
        result = demo_result()
        output = self.root / "unreloadable"
        with patch("recursive_mirror.runtime.RecursiveMirror.run", return_value=result):
            with patch("recursive_mirror.runtime.MAX_JSON_BYTES", 1):
                with self.assertRaisesRegex(ValueError, "reload limit"):
                    execute(ScriptedTestDouble(), demo_plan(), output)
        for name in ("result.json", "checkpoint.json", "report.md"):
            self.assertFalse((output / name).exists())
        events = [json.loads(x) for x in (output / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(events[-1]["event"], "runtime_failed")
        self.assertFalse(any(e["event"] == "runtime_completed" for e in events))

    def test_plan_rejects_unknown_fields_and_overlap(self):
        obj = json.loads(json.dumps(asdict(demo_plan())))
        obj["rounds"][0]["discovery"][0]["hidden_instruction"] = "ignore grading"
        with self.assertRaises(ValueError):
            plan_from_dict(obj)
        obj = json.loads(json.dumps(asdict(demo_plan())))
        obj["final_tasks"][0] = obj["rounds"][0]["discovery"][0]
        with self.assertRaises(ValueError):
            plan_from_dict(obj)

    def test_duplicate_keys_and_nonfinite_numbers_are_rejected(self):
        for raw in ('{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}'):
            path = self.root / "bad.json"
            path.write_text(raw, encoding="utf-8")
            with self.assertRaises(ValueError):
                read_json(path)

    def test_hard_backend_failure_keeps_partial_response_and_publishes_nothing(self):
        class Broken:
            identity = "scripted-failure-test-double"

            def complete(self, request):
                raise GenerationFailure("input overflow", "partial reply")

        output = self.root / "failed"
        with self.assertRaises(GenerationFailure):
            execute(Broken(), demo_plan(), output)
        for name in ("result.json", "checkpoint.json", "report.md"):
            self.assertFalse((output / name).exists())
        events = [json.loads(x) for x in (output / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(e["event"] == "run_failed" for e in events))
        self.assertTrue(any(e["event"] == "llm_error" and e.get("partial_output") == "partial reply" for e in events))

    def test_backend_is_lazy_offline_and_deterministic_by_default(self):
        backend = HuggingFaceBackend("example/model")
        self.assertFalse(backend.allow_download)
        self.assertIsNone(backend._tokenizer)
        self.assertIsNone(backend._model)
        self.assertTrue(backend.deterministic)
        self.assertTrue(backend.identity.endswith("|dtype=auto|pure_greedy|input_limit=8192"))
        self.assertIn("|input_limit=4096", HuggingFaceBackend("example/model", max_input_tokens=4096).identity)
        self.assertIn("|dtype=bfloat16|", HuggingFaceBackend("example/model", dtype="bfloat16").identity)
        for bad in ({"device": "cpu", "four_bit": True}, {"max_input_tokens": 100}, {"device": "tpu"},
                    {"dtype": "int8"}, {"device": "cuda", "four_bit": True, "dtype": "float32"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                HuggingFaceBackend("example/model", **bad)

    def test_remote_downloads_require_immutable_revision(self):
        for revision in (None, "main", "a" * 39, "A" * 40, True):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                HuggingFaceBackend("example/model", allow_download=True, revision=revision)
        revision = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
        backend = HuggingFaceBackend("example/model", allow_download=True, revision=revision)
        self.assertIn(f"revision={revision}", backend.identity)
        self.assertEqual(backend.revision, revision)
        local = self.root / "local-model"
        local.mkdir()
        self.assertIsNone(HuggingFaceBackend(str(local), allow_download=True).revision)

    def test_both_loaders_receive_same_revision_and_access_policy(self):
        revision = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
        for allow_download in (False, True):
            with self.subTest(allow_download=allow_download):
                fake_transformers = Mock()
                fake_torch = Mock()
                fake_torch.cuda.is_available.return_value = False
                backend = HuggingFaceBackend("example/model", revision=revision,
                                             allow_download=allow_download, device="cpu")
                with patch.dict("sys.modules", {"transformers": fake_transformers, "torch": fake_torch}):
                    backend._load_tokenizer()
                    backend._load_model()
                for loader in (fake_transformers.AutoTokenizer, fake_transformers.AutoModelForCausalLM):
                    self.assertEqual(loader.from_pretrained.call_args.kwargs["revision"], revision)
                    self.assertEqual(loader.from_pretrained.call_args.kwargs["local_files_only"], not allow_download)
                    self.assertFalse(loader.from_pretrained.call_args.kwargs["trust_remote_code"])

    def fake_backend(self, *, input_length=3, tokens=(10, 11), eos=99, **kwargs):
        backend = HuggingFaceBackend("fake/model", **kwargs)
        input_tensor = Mock()
        input_tensor.shape = (1, input_length)
        input_tensor.to.return_value = input_tensor
        backend._tokenizer = Mock()
        backend._tokenizer.return_value = {"input_ids": input_tensor, "token_type_ids": input_tensor}
        backend._tokenizer.decode.return_value = "preserved partial text"
        backend._tokenizer.eos_token_id = eos
        generated = Mock()
        generated.__getitem__ = Mock(return_value=Mock(tolist=Mock(return_value=list(tokens))))
        backend._model = Mock()
        backend._model.generate.return_value = generated
        backend._model.generation_config.eos_token_id = eos
        backend._model.config._commit_hash = "a" * 40
        backend._model.dtype = "torch.float32"
        backend._torch = Mock()
        backend._torch.inference_mode.return_value = nullcontext()
        backend._device = "cpu"
        return backend

    def test_input_ceiling_refuses_generation_without_truncating(self):
        backend = self.fake_backend(input_length=8193)
        with self.assertRaisesRegex(GenerationFailure, "No truncation applied"):
            backend.complete(Request("answer", "system", "prompt", 2, 42))
        backend._model.generate.assert_not_called()
        self.assertFalse(backend._tokenizer.call_args.kwargs["truncation"])
        self.assertEqual(backend.last_receipt["input_tokens"], 8193)
        self.assertEqual(backend.last_receipt["output_tokens"], 0)
        self.assertEqual(self.fake_backend(input_length=8192, tokens=(10, 99)).complete(
            Request("answer", "system", "prompt", 2, 42)), "preserved partial text")

    def test_token_limit_raises_truncation_with_partial_output_and_eos_is_accepted(self):
        backend = self.fake_backend(tokens=(10, 11))
        with self.assertRaises(OutputTruncated) as caught:
            backend.complete(Request("answer", "system", "prompt", 2, 42))
        self.assertNotIsInstance(caught.exception, GenerationFailure)
        self.assertEqual(caught.exception.partial_output, "preserved partial text")
        self.assertTrue(backend.last_receipt["output_truncated"])
        backend = self.fake_backend(tokens=(10, 99))
        self.assertEqual(backend.complete(Request("answer", "system", "prompt", 2, 42)), "preserved partial text")
        decoding = backend._model.generate.call_args.kwargs
        self.assertEqual((decoding["do_sample"], decoding["repetition_penalty"], decoding["num_beams"]), (False, 1.0, 1))
        self.assertEqual(sorted(backend._model.generate.call_args.kwargs.keys() & {"input_ids", "token_type_ids"}),
                         ["input_ids"])
        receipt = dict(backend.last_receipt)
        self.assertGreaterEqual(receipt.pop("generation_seconds"), 0)
        self.assertEqual(receipt, {
            "input_tokens": 3, "output_tokens": 2, "requested_revision": None,
            "resolved_model_commit": "a" * 40, "actual_device": "cpu",
            "actual_dtype": "torch.float32", "weight_quantization": None,
            "decoding": "pure_greedy", "repetition_penalty": 1.0,
            "input_truncated": False, "output_truncated": False, "weight_bytes_verified": False,
        })

    def test_cli_passes_revision_and_experiment_options_to_backend_and_config(self):
        plan = self.write_json("plan.json", asdict(demo_plan()))
        revision = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
        stdout = io.StringIO()
        with patch("recursive_mirror.cli.HuggingFaceBackend", return_value=ScriptedTestDouble()) as factory, \
                patch("recursive_mirror.cli.execute", wraps=execute) as run:
            with redirect_stdout(stdout):
                code = main(["run", "--plan", str(plan), "--output", str(self.root / "pinned"),
                             "--model", "example/model", "--revision", revision, "--allow-download",
                             "--candidates", "repair,rewrite", "--max-depth", "2", "--max-calls", "300",
                             "--hide-discovery-answers", "--max-input-tokens", "4096", "--max-p-value", "0.3",
                             "--dtype", "bfloat16"])
        self.assertEqual(code, 0)
        self.assertEqual(factory.call_args.kwargs["revision"], revision)
        self.assertTrue(factory.call_args.kwargs["allow_download"])
        self.assertEqual(factory.call_args.kwargs["max_input_tokens"], 4096)
        self.assertEqual(factory.call_args.kwargs["dtype"], "bfloat16")
        config = run.call_args.args[3]
        self.assertEqual((config.candidate_kinds, config.max_depth, config.max_calls, config.reveal_discovery_answers,
                          config.max_p_value), (("repair", "rewrite"), 2, 300, False, 0.3))
        self.assertIsNone(run.call_args.args[4])
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["accepted_levels"], 1)
        self.assertTrue(summary["transfer_improved"])

    def test_cli_initial_strategy_sets_a_control_start_and_excludes_state(self):
        plan = self.write_json("plan.json", asdict(demo_plan()))
        with patch("recursive_mirror.cli.HuggingFaceBackend", return_value=ScriptedTestDouble()), \
                patch("recursive_mirror.cli.execute", wraps=execute) as run, redirect_stdout(io.StringIO()):
            self.assertEqual(main(["run", "--plan", str(plan), "--output", str(self.root / "control"),
                                   "--model", "m", "--initial-strategy", "Think step by step."]), 0)
        self.assertEqual(run.call_args.args[4], State(strategy="Think step by step."))
        checkpoint = self.write_json("checkpoint.json", checkpoint_for(demo_result()))
        with patch("recursive_mirror.cli.HuggingFaceBackend", return_value=ScriptedTestDouble()), \
                redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(main(["run", "--plan", str(plan), "--output", str(self.root / "both"),
                                   "--model", "m", "--initial-strategy", "x", "--state", str(checkpoint)]), 1)
        self.assertIn("cannot be combined", stderr.getvalue())
        self.assertFalse((self.root / "both").exists())

    def test_cli_demo_is_explicit_and_refuses_reuse(self):
        output = self.root / "cli-demo"
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(main(["demo", "--output", str(output)]), 0)
            self.assertEqual(main(["demo", "--output", str(output)]), 1)
        self.assertIn("software-test evidence only", stdout.getvalue())
        self.assertIn("scripted-test-double", stdout.getvalue())
        self.assertIn("FileExistsError", stderr.getvalue())

    def test_report_names_each_outcome_honestly(self):
        result = demo_result()
        no_level = replace(result, steps=(), status="call_budget", transfer_improved=False)
        self.assertIn("no improvement level ran (stop reason `call_budget`)", render_report(no_level))
        rolled_back = replace(result, transfer_improved=False)
        self.assertIn("did not transfer to the held-out final tasks", render_report(rolled_back))
        calibration = replace(result, final_comparison=replace(result.final_comparison, reason="calibration_gain"))
        self.assertIn("forecast its own success better on held-out final tasks", render_report(calibration))

    def test_continuation_with_a_fresh_excluded_plan_and_refusal_before_any_directory(self):
        first = self.root / "first"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["demo", "--output", str(first)]), 0)
        checkpoint = first / "checkpoint.json"
        plan_path = self.root / "plans" / "next.json"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main(["make-plan", "--output", str(plan_path), "--seed", "9", "--rounds", "1",
                                   "--discovery", "2", "--validation", "2", "--final", "2",
                                   "--exclude-state", str(checkpoint)]), 0)
        used = {o.prompt_sha256 for o in load_checkpoint(checkpoint).observations}
        self.assertFalse({fingerprint(t.prompt) for t in load_plan(plan_path).tasks()} & used)
        stale_plan = self.write_json("stale.json", asdict(demo_plan()))
        output = self.root / "refused"
        with patch("recursive_mirror.cli.HuggingFaceBackend", return_value=ScriptedTestDouble()), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(main(["run", "--plan", str(stale_plan), "--output", str(output), "--model", "m",
                                   "--state", str(checkpoint)]), 1)
        self.assertIn("discover-1 was already used", stderr.getvalue())
        self.assertFalse(output.exists())

    def test_cli_make_plan_writes_a_loadable_plan_and_refuses_overwrite(self):
        target = self.root / "plan.json"
        args = ["make-plan", "--output", str(target), "--seed", "5", "--rounds", "1", "--discovery", "2",
                "--validation", "3", "--final", "4", "--categories", "weekday,letter_count"]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(main(args), 0)
            self.assertEqual(main(args), 1)
        self.assertIn("FileExistsError", stderr.getvalue())
        plan = load_plan(target)
        self.assertEqual([len(plan.rounds[0].discovery), len(plan.rounds[0].validation), len(plan.final_tasks)],
                         [2, 3, 4])
        self.assertEqual({t.category for t in plan.tasks()}, {"weekday", "letter_count"})
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["make-plan", "--output", str(self.root / "x.json"), "--categories", "poetry"])


if __name__ == "__main__":
    unittest.main()
