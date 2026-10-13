"""Real torch/transformers path with a tiny locally built Qwen2 model; no download.

Skipped unless torch, transformers and tokenizers are importable. The Colab
notebook runs this module on the GPU before the long trial, so a backend or
library incompatibility fails in seconds instead of mid-run.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

from recursive_mirror.cli import main
from recursive_mirror.core import Config, OutputTruncated, Plan, Request, Round, Task, extract_answer
from recursive_mirror.runtime import HuggingFaceBackend, execute, load_checkpoint


HAVE_STACK = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "tokenizers"))
DEVICE = os.environ.get("RECURSIVE_MIRROR_TEST_DEVICE", "cpu")
CHAT_TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
                 "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def build_tiny_qwen(directory: Path, *, eos_first: bool) -> None:
    """A 2-layer Qwen2 with a trained byte-level BPE tokenizer and Qwen's chat format.

    eos_first=True zeroes the output head so greedy decoding emits token 0
    (<|endoftext|>, an EOS id) immediately; otherwise random weights rarely
    stop, which exercises the truncation path.
    """
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GenerationConfig, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    corpus = ["Strategy self-model forecast answer task category reflection ANSWER: 42",
              "What is 441 multiplied by 94? Give the exact integer without separators."] * 20
    raw = Tokenizer(models.BPE())
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    raw.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=400, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(directory)
    ids = [tokenizer.convert_tokens_to_ids(t) for t in ("<|endoftext|>", "<|im_end|>")]
    config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=16384,
                         eos_token_id=ids, bos_token_id=ids[0], tie_word_embeddings=False)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config)
    if eos_first:
        with torch.no_grad():
            model.lm_head.weight.zero_()
    # Mirror Qwen2.5's shipped sampling defaults, which greedy decoding must override.
    model.generation_config = GenerationConfig(eos_token_id=ids, pad_token_id=ids[0], do_sample=True,
                                               temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05)
    model.save_pretrained(directory, safe_serialization=True)


def small_plan() -> Plan:
    def task(name, a, b):
        return Task(name, f"What is {a} multiplied by {b}? Give the exact integer without separators.", str(a * b),
                    "multiplication")
    return Plan((Round((task("d1", 12, 13),), (task("v1", 14, 15), task("v2", 16, 17))),),
                (task("f1", 18, 19), task("f2", 21, 22)))


@unittest.skipUnless(HAVE_STACK, "torch/transformers/tokenizers not installed")
class HuggingFaceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.random_model = cls.root / "tiny-random"
        cls.eos_model = cls.root / "tiny-eos"
        build_tiny_qwen(cls.random_model, eos_first=False)
        build_tiny_qwen(cls.eos_model, eos_first=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def backend(self, path, **kwargs):
        return HuggingFaceBackend(str(path), device=DEVICE, **kwargs)

    def test_eos_reply_has_receipt_and_is_deterministic(self):
        backend = self.backend(self.eos_model)
        request = Request("answer", "Solve it.", "What is 2 + 2?", 16, 7)
        first = backend.complete(request)
        self.assertEqual(first, backend.complete(request))
        receipt = backend.last_receipt
        self.assertEqual(receipt["output_tokens"], 1)
        self.assertFalse(receipt["output_truncated"])
        self.assertGreater(receipt["input_tokens"], 5)
        self.assertEqual(receipt["actual_device"], "cuda:0" if DEVICE == "cuda" else "cpu")
        self.assertTrue(backend.deterministic)

    def test_token_limit_raises_truncation_with_partial_text(self):
        backend = self.backend(self.random_model)
        with self.assertRaises(OutputTruncated) as caught:
            backend.complete(Request("perspective:actor", "Reflect.", "Evidence.", 12, 3))
        self.assertTrue(caught.exception.partial_output)
        self.assertEqual(backend.last_receipt["output_tokens"], 12)
        self.assertTrue(backend.last_receipt["output_truncated"])
        self.assertEqual(extract_answer(caught.exception.partial_output)[1] in ("answer_line", "last_line", "empty"), True)

    def test_input_limit_is_enforced_with_the_real_tokenizer(self):
        backend = self.backend(self.eos_model, max_input_tokens=256)
        with self.assertRaisesRegex(Exception, "No truncation applied"):
            backend.complete(Request("answer", "Solve it.", "word " * 2000, 4, 1))
        self.assertGreater(backend.last_receipt["input_tokens"], 256)

    def test_full_recursive_run_survives_real_truncations_and_publishes_results(self):
        output = self.root / "run-random"
        config = Config(max_depth=1, max_calls=200, forecast_tokens=4, answer_tokens=8, reflection_tokens=8,
                        revision_tokens=8)
        result = execute(self.backend(self.random_model), small_plan(), output, config)
        self.assertEqual(result.status, "plan_exhausted")
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(load_checkpoint(output / "checkpoint.json"), result.committed_state)
        events = [json.loads(line) for line in (output / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(e["event"] == "llm_truncated" for e in events))
        self.assertEqual(events[-1]["event"], "runtime_completed")
        self.assertTrue((output / "report.md").read_text(encoding="utf-8").startswith("# Recursive Mirror run report"))
        # Greedy replies are replayed for byte-identical requests (the unchanged final arm).
        self.assertGreater(result.cache_hits, 0)

    def test_cli_run_with_a_local_model_directory(self):
        plan = self.root / "plan.json"
        plan.write_text(json.dumps(asdict(small_plan())), encoding="utf-8")
        output = self.root / "cli-run"
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            code = main(["run", "--plan", str(plan), "--output", str(output), "--model", str(self.eos_model),
                         "--device", DEVICE, "--max-depth", "1", "--max-calls", "100", "--answer-tokens", "8"])
        self.assertEqual(code, 0, stdout.getvalue())
        summary = json.loads(stdout.getvalue())
        self.assertIn("huggingface:", summary["model_identity"])
        self.assertFalse(summary["transfer_improved"])


if __name__ == "__main__":
    unittest.main()
