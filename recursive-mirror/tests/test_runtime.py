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
from recursive_mirror.core import Config, Perspective, RecursiveMirror, Request, State, canonical, digest
from recursive_mirror.demo import ScriptedTestDouble, demo_plan
from recursive_mirror.runtime import (GenerationFailure, HuggingFaceBackend, checkpoint_for,
                                     execute, load_checkpoint, load_plan, plan_from_dict, read_json)


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

    def test_demo_persists_recursive_events_and_committed_state(self):
        output = self.root / "demo"
        result = execute(ScriptedTestDouble(), demo_plan(), output, Config(max_depth=2, max_calls=64))
        self.assertTrue(result.transfer_improved)
        self.assertEqual([s.accepted for s in result.steps], [True, False])
        self.assertEqual(result.model_identity, "scripted-test-double")
        self.assertEqual(load_checkpoint(output / "checkpoint.json"), result.committed_state)
        events = [json.loads(line) for line in (output / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(sum(e["event"] == "recursive_enter" for e in events), 2)
        self.assertEqual(sum(e["event"] == "llm_response" for e in events), result.calls)
        self.assertEqual(load_plan(output / "plan.json"), demo_plan())
        self.assertEqual(events[-1]["event"], "runtime_completed")

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
        result = RecursiveMirror(ScriptedTestDouble(), demo_plan()).run()
        original = checkpoint_for(result)
        mutations = [
            lambda obj: obj["state"].update(version=True),
            lambda obj: obj["state"]["observations"][0].update(correct="false"),
            lambda obj: obj["state"]["observations"][0].update(probability=True),
            lambda obj: obj["state"]["perspectives"][0].update(round_index=False),
            lambda obj: obj["state"]["perspectives"][0].update(angle=[]),
            lambda obj: obj["state"]["observations"][0].update(probability=0.1),
            lambda obj: obj["state"].update(extra="injected"),
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
        result = RecursiveMirror(ScriptedTestDouble(), demo_plan()).run()
        view = Perspective(0, "actor", "a" * 64, "Retained perspective")
        for field, item in (("perspectives", view), ("observations", result.committed_state.observations[0])):
            with self.subTest(field=field):
                state = State(perspectives=(), observations=())
                at_limit = replace(result, committed_state=replace(state, **{field: (item,) * 2}))
                beyond = replace(result, committed_state=replace(state, **{field: (item,) * 3}))
                # Exercise the same shared cap at a small bound, preserving all
                # entries. The production constant is 10,000 for either history.
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
        result = RecursiveMirror(ScriptedTestDouble(), demo_plan()).run()
        result = replace(result, committed_state=State(strategy="Unicode strategy: \u03c0", perspectives=(), observations=()))
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

    def test_checkpoint_continuation_requires_same_model_identity(self):
        result = RecursiveMirror(ScriptedTestDouble(), demo_plan()).run()
        path = self.write_json("checkpoint.json", checkpoint_for(result))
        self.assertEqual(load_checkpoint(path, expected_model_identity="scripted-test-double"), result.committed_state)
        with self.assertRaisesRegex(ValueError, "model identity"):
            load_checkpoint(path, expected_model_identity="different-model-or-revision")
        self.assertEqual(load_checkpoint(path), result.committed_state)

    def test_cli_rejects_cross_model_state_before_loading_or_creating_output(self):
        result = RecursiveMirror(ScriptedTestDouble(), demo_plan()).run()
        checkpoint = self.write_json("checkpoint.json", checkpoint_for(result))
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
        result = RecursiveMirror(ScriptedTestDouble(), demo_plan()).run()
        output = self.root / "unreloadable"
        with patch("recursive_mirror.runtime.RecursiveMirror.run", return_value=result):
            with patch("recursive_mirror.runtime.MAX_JSON_BYTES", 1):
                with self.assertRaisesRegex(ValueError, "reload limit"):
                    execute(ScriptedTestDouble(), demo_plan(), output)
        self.assertFalse((output / "result.json").exists())
        self.assertFalse((output / "checkpoint.json").exists())
        events = [json.loads(x) for x in (output / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(events[-1]["event"], "runtime_failed")
        self.assertFalse(any(e["event"] == "runtime_completed" for e in events))

    def test_plan_rejects_unknown_fields_and_overlap(self):
        obj = asdict(demo_plan())
        obj = json.loads(json.dumps(obj))
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

    def test_partial_response_survives_failed_run(self):
        class Broken:
            identity = "scripted-failure-test-double"

            def complete(self, request):
                raise GenerationFailure("token limit", "partial reply")

        output = self.root / "failed"
        with self.assertRaises(GenerationFailure):
            execute(Broken(), demo_plan(), output)
        self.assertFalse((output / "result.json").exists())
        self.assertFalse((output / "checkpoint.json").exists())
        events = [json.loads(x) for x in (output / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(e["event"] == "run_failed" for e in events))
        self.assertTrue(any(e["event"] == "llm_error" and e.get("partial_output") == "partial reply" for e in events))

    def test_backend_is_lazy_and_offline_by_default(self):
        backend = HuggingFaceBackend("example/model")
        self.assertFalse(backend.allow_download)
        self.assertIsNone(backend._tokenizer)
        self.assertIsNone(backend._model)
        with self.assertRaises(ValueError):
            HuggingFaceBackend("example/model", device="cpu", four_bit=True)

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

    def fake_backend(self, *, input_length=3, tokens=(10, 11), eos=99):
        backend = HuggingFaceBackend("fake/model")
        input_tensor = Mock()
        input_tensor.shape = (1, input_length)
        input_tensor.to.return_value = input_tensor
        backend._tokenizer = Mock()
        backend._tokenizer.return_value = {"input_ids": input_tensor}
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
        backend = self.fake_backend(input_length=4097)
        with self.assertRaisesRegex(GenerationFailure, "No truncation applied"):
            backend.complete(Request("answer", "system", "prompt", 2, 42))
        backend._model.generate.assert_not_called()
        self.assertFalse(backend._tokenizer.call_args.kwargs["truncation"])
        self.assertEqual(backend.last_receipt["input_tokens"], 4097)
        self.assertEqual(backend.last_receipt["output_tokens"], 0)

    def test_token_limit_retains_partial_output_and_eos_is_accepted(self):
        backend = self.fake_backend(tokens=(10, 11))
        with self.assertRaises(GenerationFailure) as caught:
            backend.complete(Request("answer", "system", "prompt", 2, 42))
        self.assertEqual(caught.exception.partial_output, "preserved partial text")
        self.assertTrue(backend.last_receipt["output_truncated"])
        backend = self.fake_backend(tokens=(10, 99))
        self.assertEqual(backend.complete(Request("answer", "system", "prompt", 2, 42)), "preserved partial text")
        self.assertFalse(backend._model.generate.call_args.kwargs["do_sample"])
        self.assertEqual(backend.last_receipt, {
            "input_tokens": 3, "output_tokens": 2, "requested_revision": None,
            "resolved_model_commit": "a" * 40, "actual_device": "cpu",
            "actual_dtype": "torch.float32", "weight_quantization": None,
            "input_truncated": False, "output_truncated": False, "weight_bytes_verified": False,
        })

    def test_cli_passes_revision_to_backend(self):
        plan = self.write_json("plan.json", asdict(demo_plan()))
        revision = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
        stdout = io.StringIO()
        with patch("recursive_mirror.cli.HuggingFaceBackend", return_value=ScriptedTestDouble()) as factory:
            with redirect_stdout(stdout):
                code = main(["run", "--plan", str(plan), "--output", str(self.root / "pinned"),
                             "--model", "example/model", "--revision", revision, "--allow-download"])
        self.assertEqual(code, 0)
        self.assertEqual(factory.call_args.kwargs["revision"], revision)
        self.assertTrue(factory.call_args.kwargs["allow_download"])

    def test_cli_demo_is_explicit_and_refuses_reuse(self):
        output = self.root / "cli-demo"
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(main(["demo", "--output", str(output)]), 0)
            self.assertEqual(main(["demo", "--output", str(output)]), 1)
        self.assertIn("software-test evidence only", stdout.getvalue())
        self.assertIn("scripted-test-double", stdout.getvalue())
        self.assertIn("FileExistsError", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
