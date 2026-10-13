"""Local persistence, run reports and an optional, lazily loaded Hugging Face backend."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable

from .core import (ANSWER_STATUSES, OUTCOMES, PHASES, Attempt, Config, LanguageModel, Observation,
                   OutputTruncated, Perspective, Plan, Request, RecursiveMirror, Result, Round, State,
                   Task, canonical, digest, excerpt, parse_forecast)


CHECKPOINT_FORMAT = "recursive-mirror-checkpoint-v2"
LEGACY_CHECKPOINT_FORMATS = ("recursive-mirror-checkpoint-v1",)
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_COLLECTION_ITEMS = 10000
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _bad_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON number: {value}")


def read_json(path: str | Path, max_bytes: int | None = None) -> object:
    max_bytes = MAX_JSON_BYTES if max_bytes is None else max_bytes
    with Path(path).open("rb") as handle:
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("JSON document exceeds its declared size limit")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_object_pairs,
                      parse_constant=_bad_constant)


def _keys(value: object, required: set[str], optional: set[str] | None = None) -> dict:
    if type(value) is not dict or not required <= value.keys() or value.keys() - required - (optional or set()):
        raise ValueError(f"Expected an object with required keys {sorted(required)}")
    return value


def _string(value: object, label: str, limit: int = 12000, *, blank_ok: bool = False) -> str:
    if type(value) is not str or len(value) > limit or (not blank_ok and not value.strip()):
        raise ValueError(f"{label} must be {'text' if blank_ok else 'nonblank text'} of at most {limit} characters")
    return value


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _unit(value: object, label: str) -> float | None:
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f"{label} must be null or a number in [0, 1]")
    return value


def _sha(value: object, label: str) -> str:
    if type(value) is not str or not _SHA.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _array(value: object, label: str, *, nonempty: bool = False) -> list:
    if type(value) is not list or len(value) > MAX_COLLECTION_ITEMS or (nonempty and not value):
        raise ValueError(f"{label} must be {'a nonempty' if nonempty else 'an'} array of at most {MAX_COLLECTION_ITEMS} items")
    return value


def plan_from_dict(value: object) -> Plan:
    obj = _keys(value, {"rounds", "final_tasks"})

    def tasks(items: object) -> tuple[Task, ...]:
        result = []
        for item in _array(items, "tasks", nonempty=True):
            row = _keys(item, {"task_id", "prompt", "expected"}, {"category"})
            result.append(Task(**row))
        return tuple(result)

    rounds = []
    for item in _array(obj["rounds"], "rounds", nonempty=True):
        row = _keys(item, {"discovery", "validation"})
        rounds.append(Round(tasks(row["discovery"]), tasks(row["validation"])))
    return Plan(tuple(rounds), tasks(obj["final_tasks"]))


def load_plan(path: str | Path) -> Plan:
    return plan_from_dict(read_json(path))


_V1_OBSERVATION = {"round_index", "phase", "task_id", "category", "forecast_raw",
                   "probability", "answer", "correct", "prompt_sha256"}
_V2_OBSERVATION = _V1_OBSERVATION | {"policy_sha256", "arm", "answer_status"}
_ATTEMPT = {"round_index", "kind", "strategy", "self_model", "outcome", "reason", "n",
            "parent_accuracy", "candidate_accuracy", "parent_brier", "candidate_brier", "wins", "losses"}


def _perspective(item: object, legacy: bool) -> Perspective:
    required = {"round_index", "angle", "state_sha256", "text"}
    row = dict(_keys(item, required if legacy else required | {"complete"}))
    _integer(row["round_index"], "perspective round_index")
    if type(row["angle"]) is not str or row["angle"] not in RecursiveMirror.ANGLES:
        raise ValueError("Unknown perspective angle")
    _sha(row["state_sha256"], "perspective state_sha256")
    row.setdefault("complete", True)
    if type(row["complete"]) is not bool:
        raise ValueError("Perspective complete must be a boolean")
    _string(row["text"], "perspective text", blank_ok=not row["complete"])
    return Perspective(**row)


def _observation(item: object, legacy: bool) -> Observation:
    row = dict(_keys(item, _V1_OBSERVATION if legacy else _V2_OBSERVATION))
    if legacy:
        row.update(policy_sha256="", arm="", answer_status="legacy")
    _integer(row["round_index"], "observation round_index", -1)
    if type(row["phase"]) is not str or row["phase"] not in PHASES:
        raise ValueError("Unknown observation phase")
    if (row["round_index"] == -1) != row["phase"].startswith("final_"):
        raise ValueError("Observation phase and round_index disagree")
    for key in ("task_id", "category"):
        _string(row[key], key)
    _string(row["forecast_raw"], "forecast_raw", blank_ok=True)
    _string(row["answer"], "answer", blank_ok=True)
    _sha(row["prompt_sha256"], "observation prompt_sha256")
    if row["policy_sha256"] != "":
        _sha(row["policy_sha256"], "observation policy_sha256")
    if type(row["arm"]) is not str or (row["arm"] not in RecursiveMirror.KINDS and
                                       not (row["arm"] == "" and (legacy or row["phase"] != "validation_candidate"))):
        raise ValueError("Observation arm disagrees with its phase")
    if row["arm"] and row["phase"] != "validation_candidate":
        raise ValueError("Only validation candidates carry an arm")
    if type(row["answer_status"]) is not str or row["answer_status"] not in ANSWER_STATUSES:
        raise ValueError("Unknown observation answer_status")
    probability = _unit(row["probability"], "Observation probability")
    # A recorded probability must be exactly what the raw forecast parses to.
    if probability is not None and probability != parse_forecast(row["forecast_raw"]):
        raise ValueError("Observation probability disagrees with its raw forecast")
    if type(row["correct"]) is not bool or (row["correct"] and not row["answer"].strip()):
        raise ValueError("Observation correct must be a boolean backed by an answer")
    return Observation(**row)


def _attempt(item: object) -> Attempt:
    row = _keys(item, _ATTEMPT)
    _integer(row["round_index"], "attempt round_index", -1)
    kinds = (*RecursiveMirror.KINDS, "final_transfer")
    if type(row["kind"]) is not str or row["kind"] not in kinds:
        raise ValueError("Unknown attempt kind")
    if (row["round_index"] == -1) != (row["kind"] == "final_transfer"):
        raise ValueError("Attempt kind and round_index disagree")
    if type(row["outcome"]) is not str or row["outcome"] not in OUTCOMES:
        raise ValueError("Unknown attempt outcome")
    _string(row["reason"], "attempt reason", 200)
    _string(row["strategy"], "attempt strategy", blank_ok=True)
    _string(row["self_model"], "attempt self_model", blank_ok=True)
    for key in ("n", "wins", "losses"):
        _integer(row[key], f"attempt {key}")
    for key in ("parent_accuracy", "candidate_accuracy", "parent_brier", "candidate_brier"):
        _unit(row[key], f"attempt {key}")
    return Attempt(**row)


def state_from_dict(value: object, *, legacy: bool = False) -> State:
    base = {"strategy", "self_model", "version", "perspectives", "observations"}
    obj = _keys(value, base if legacy else base | {"attempts"})
    _string(obj["strategy"], "strategy")
    _string(obj["self_model"], "self_model")
    _integer(obj["version"], "version")
    perspectives = tuple(_perspective(item, legacy) for item in _array(obj["perspectives"], "perspectives"))
    observations = tuple(_observation(item, legacy) for item in _array(obj["observations"], "observations"))
    attempts = () if legacy else tuple(_attempt(item) for item in _array(obj["attempts"], "attempts"))
    return State(obj["strategy"], obj["self_model"], obj["version"], perspectives, observations, attempts)


def checkpoint_for(result: Result) -> dict:
    payload = {"format": CHECKPOINT_FORMAT, "model_identity": result.model_identity,
               "plan_sha256": result.plan_sha256, "state": asdict(result.committed_state),
               "state_sha256": result.committed_state.sha256}
    checkpoint = {**payload, "checkpoint_sha256": digest(payload)}
    # Validate exactly the JSON representation that will be written, including
    # tuple-to-array conversion, UTF-8 bytes and the final newline. Never publish
    # a successful checkpoint that its paired loader would refuse to restore.
    encoded = canonical(checkpoint) + "\n"
    if len(encoded.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError(f"Checkpoint exceeds the {MAX_JSON_BYTES}-byte reload limit; save aborted without trimming history")
    serialized = json.loads(encoded)
    _checkpoint_state(serialized)
    return serialized


def _checkpoint_state(value: object) -> State:
    obj = _keys(value, {"format", "model_identity", "plan_sha256", "state",
                        "state_sha256", "checkpoint_sha256"})
    if obj["format"] != CHECKPOINT_FORMAT and obj["format"] not in LEGACY_CHECKPOINT_FORMATS:
        raise ValueError("Unsupported checkpoint format")
    _string(obj["model_identity"], "model_identity")
    for field in ("plan_sha256", "state_sha256", "checkpoint_sha256"):
        _sha(obj[field], field)
    payload = {key: value for key, value in obj.items() if key != "checkpoint_sha256"}
    if digest(payload) != obj["checkpoint_sha256"]:
        raise ValueError("Checkpoint content hash mismatch")
    legacy = obj["format"] in LEGACY_CHECKPOINT_FORMATS
    if legacy:
        # Verify the v1 state hash on its original representation, then upgrade.
        if digest(obj["state"]) != obj["state_sha256"]:
            raise ValueError("Checkpoint state hash mismatch")
        return state_from_dict(obj["state"], legacy=True)
    state = state_from_dict(obj["state"])
    if state.sha256 != obj["state_sha256"]:
        raise ValueError("Checkpoint state hash mismatch")
    return state


def load_checkpoint(path: str | Path, *, expected_model_identity: str | None = None) -> State:
    value = read_json(path)
    state = _checkpoint_state(value)
    if expected_model_identity is not None:
        _string(expected_model_identity, "expected_model_identity")
        if value["model_identity"] != expected_model_identity:
            raise ValueError("Checkpoint model identity does not match the requested backend; continuation refused")
    return state


def _save_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value if isinstance(value, str) else canonical(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_report(result: Result) -> str:
    """A human-readable summary of what the run measured. The journal stays authoritative."""
    final = result.final_comparison
    lines = ["# Recursive Mirror run report", "",
             f"- Protocol: `{result.protocol}`",
             f"- Model: `{result.model_identity}`",
             f"- Stop reason: `{result.status}`; LLM calls: {result.calls} (plus {result.cache_hits} "
             "deterministic cache hits)",
             f"- Levels completed: {len(result.steps)}; accepted improvements: "
             f"{sum(s.accepted for s in result.steps)}"]
    if not result.steps:
        lines.append(f"- **Outcome: no improvement level ran (stop reason `{result.status}`); only the "
                     "initial policy was evaluated.**")
    elif result.transfer_improved and final.reason == "accuracy_gain":
        lines.append(f"- **Outcome: the self-improved policy answered more held-out final tasks correctly "
                     f"({_pct(final.parent.accuracy)} -> {_pct(final.candidate.accuracy)}) and was committed.**")
    elif result.transfer_improved:
        lines.append(f"- **Outcome: the self-improved policy forecast its own success better on held-out "
                     f"final tasks (Brier {_num(final.parent.brier)} -> {_num(final.candidate.brier)}; accuracy "
                     f"{_pct(final.parent.accuracy)} -> {_pct(final.candidate.accuracy)}) and was committed.**")
    elif any(s.accepted for s in result.steps):
        lines.append("- **Outcome: a policy improved on validation but did not transfer to the held-out "
                     "final tasks; the committed policy was rolled back to the initial one.**")
    else:
        lines.append("- **Outcome: no proposal passed validation; the initial policy remains committed.**")
    lines += ["", f"## Final held-out comparison (n = {final.parent.n})", "",
              "| policy | accuracy | Brier | forecast coverage | mean forecast |",
              "|---|---|---|---|---|"]
    for name, m in (("initial", final.parent), ("self-improved candidate", final.candidate)):
        lines.append(f"| {name} | {_pct(m.accuracy)} | {_num(m.brier)} | {_pct(m.forecast_coverage)} | "
                     f"{_pct(m.mean_forecast)} |")
    lines += ["", f"Wins {final.wins}, losses {final.losses}. One-sided paired p-values: accuracy "
              f"{final.accuracy_p:.3f}, calibration {final.calibration_p:.3f} (gate requires <= "
              f"{result.config.max_p_value:g}; exploratory, not a confirmatory test). Gate: "
              f"{'passed' if final.passed else 'failed'} (`{final.reason}`).", ""]
    for step in result.steps:
        d, p = step.discovery_metrics, step.parent_metrics
        lines += [f"## Level {step.round_index}", "",
                  f"Discovery: accuracy {_pct(d.accuracy)} on {d.n} tasks, mean forecast "
                  f"{_pct(d.mean_forecast)}. Parent on validation: accuracy {_pct(p.accuracy)}, "
                  f"Brier {_num(p.brier)} (n = {p.n}).", "",
                  "| candidate | gate | accuracy | paired Brier | wins/losses | p (acc / cal) |",
                  "|---|---|---|---|---|---|"]
        for c in step.candidates:
            if c.comparison is None:
                lines.append(f"| {c.kind} | ineligible (`{c.reason}`) | | | | |")
                continue
            cmp = c.comparison
            gate = ("**selected**" if step.selected_kind == c.kind else
                    "passed" if cmp.passed else "rejected") + f" (`{cmp.reason}`)"
            lines.append(f"| {c.kind} | {gate} | {_pct(cmp.parent.accuracy)} -> {_pct(cmp.candidate.accuracy)} | "
                         f"{_num(cmp.parent_paired_brier)} -> {_num(cmp.candidate_paired_brier)} | "
                         f"{cmp.wins}/{cmp.losses} | {cmp.accuracy_p:.2f} / {cmp.calibration_p:.2f} |")
        for c in step.candidates:
            if c.strategy.strip() or c.self_model.strip():
                lines += ["", f"<details><summary>{c.kind} proposal</summary>", "",
                          "Strategy:", "", "```text", excerpt(c.strategy, 3000) or "(none)", "```", "",
                          "Self-model:", "", "```text", excerpt(c.self_model, 3000) or "(none)", "```",
                          "", "</details>"]
        lines.append("")
    committed = result.committed_state
    lines += ["## Initial policy", "", "Strategy:", "", "```text", result.initial_strategy or "(not recorded)",
              "```", "", "Self-model:", "", "```text", result.initial_self_model or "(not recorded)", "```", "",
              "## Committed policy", "", "Strategy:", "", "```text", committed.strategy, "```", "",
              "Self-model:", "", "```text", committed.self_model, "```", "",
              "## Claim boundary", "",
              "This run measures whether an inference-time strategy and self-model, revised by the "
              "model through recursive self-examination, improved exact-answer accuracy or forecast "
              "calibration on held-out tasks. Model weights were not changed. Small task sets give "
              "noisy estimates. Nothing here certifies awareness or consciousness.", ""]
    return "\n".join(lines)


class Journal:
    """Flush each event so partial runs retain requests, responses and failures."""
    def __init__(self, path: Path):
        self._handle = path.open("x", encoding="utf-8", newline="\n")

    def __call__(self, event: dict) -> None:
        self._handle.write(canonical(event) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()


def execute(model: LanguageModel, plan: Plan, output: str | Path,
            config: Config | None = None, initial: State | None = None,
            on_event: Callable[[dict], None] | None = None) -> Result:
    """Run one experiment into a new directory. on_event sees each event after it is journalled."""
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    journal = Journal(directory / "journal.jsonl")
    started = time.monotonic()

    def emit(event: dict) -> None:
        # Wall-clock offsets are journal metadata only; results and checkpoints never include them.
        journal({**event, "elapsed_seconds": round(time.monotonic() - started, 3)})
        if on_event is not None:
            try:
                on_event(event)
            except Exception:  # progress display must never alter the experiment
                pass

    try:
        _save_exclusive(directory / "plan.json", asdict(plan))
        engine = RecursiveMirror(model, plan, config, emit)
        result = engine.run(initial)
        checkpoint = checkpoint_for(result)
        report = render_report(result)
        _save_exclusive(directory / "result.json", asdict(result))
        _save_exclusive(directory / "checkpoint.json", checkpoint)
        _save_exclusive(directory / "report.md", report)
        journal({"event": "runtime_completed", "state_sha256": result.committed_state.sha256,
                 "checkpoint_sha256": checkpoint["checkpoint_sha256"]})
        return result
    except Exception as exc:
        journal({"event": "runtime_failed", "error_type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        journal.close()


class GenerationFailure(RuntimeError):
    """A hard backend failure (for example input overflow); it aborts the run."""
    def __init__(self, message: str, partial_output: str = ""):
        super().__init__(message)
        self.partial_output = partial_output


DTYPES = ("auto", "float16", "bfloat16", "float32")


def _dtype_keyword() -> str:
    """transformers >= 4.56 renamed from_pretrained(torch_dtype=...) to dtype=..."""
    try:
        import transformers
        major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
        return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    except Exception:
        return "torch_dtype"


class HuggingFaceBackend:
    """Pure greedy inference. Importing this module never imports or downloads models.

    Decoding is plain argmax: the model's shipped sampling settings and its
    repetition penalty (Qwen2.5 ships 1.05, which transformers applies even when
    do_sample=False) are overridden. It is declared deterministic, so the engine
    reuses replies to byte-identical requests within one run. Input above
    max_input_tokens is refused, never truncated. Output that reaches
    max_new_tokens without EOS is raised as OutputTruncated with its partial
    text: the engine records it and scores it as unusable instead of aborting.
    dtype "auto" means float16 on CUDA (native on T4) and float32 on CPU.
    """
    deterministic = True
    DTYPES = DTYPES

    def __init__(self, model: str, *, allow_download: bool = False,
                 device: str = "auto", four_bit: bool = False,
                 revision: str | None = None, max_input_tokens: int = 8192, dtype: str = "auto"):
        _string(model, "model")
        if revision is not None and (type(revision) is not str or not re.fullmatch(r"[0-9a-f]{40}", revision)):
            raise ValueError("revision must be a full 40-character lowercase hexadecimal commit")
        if allow_download and not Path(model).is_dir() and revision is None:
            raise ValueError("Remote downloads require an explicit immutable --revision commit")
        if device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be auto, cuda or cpu")
        if four_bit and device == "cpu":
            raise ValueError("Four-bit inference requires CUDA")
        if type(max_input_tokens) is not int or not 256 <= max_input_tokens <= 131072:
            raise ValueError("max_input_tokens must be an integer in 256..131072")
        if dtype not in self.DTYPES:
            raise ValueError(f"dtype must be one of {self.DTYPES}")
        if four_bit and dtype == "float32":
            raise ValueError("Four-bit inference computes in float16 or bfloat16")
        self.model_name = model
        self.allow_download = allow_download
        self.device_request = device
        self.four_bit = four_bit
        self.revision = revision
        self.max_input_tokens = max_input_tokens
        self.dtype = dtype
        self.identity = (f"huggingface:{model}|revision={revision or 'unresolved-local-or-cache'}"
                         f"|device={device}|four_bit={str(four_bit).lower()}|dtype={dtype}"
                         f"|pure_greedy|input_limit={max_input_tokens}")
        self.last_receipt: dict | None = None
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = None

    def _load_tokenizer(self) -> None:
        if self._tokenizer is not None:
            return
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, local_files_only=not self.allow_download,
            trust_remote_code=False, revision=self.revision)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        use_cuda = self.device_request == "cuda" or (self.device_request == "auto" and torch.cuda.is_available())
        if use_cuda and not torch.cuda.is_available():
            raise GenerationFailure("CUDA was requested but is unavailable")
        if self.four_bit and not use_cuda:
            raise GenerationFailure("Four-bit inference requires an available CUDA device")
        self._torch = torch
        self._device = "cuda:0" if use_cuda else "cpu"
        # T4 has native FP16, not native BF16, so "auto" keeps CUDA arithmetic FP16.
        name = self.dtype if self.dtype != "auto" else ("float16" if use_cuda else "float32")
        compute = getattr(torch, name)
        kwargs = {"local_files_only": not self.allow_download, "trust_remote_code": False,
                  "revision": self.revision, "use_safetensors": True, _dtype_keyword(): compute}
        if self.four_bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute)
            kwargs["device_map"] = {"": self._device}
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        if not self.four_bit:
            self._model.to(self._device)
        self._model.eval()

    def _receipt(self, input_tokens: int, output_tokens: int,
                 output_truncated: bool = False) -> dict:
        resolved = getattr(getattr(self._model, "config", None), "_commit_hash", None)
        return {"input_tokens": input_tokens, "output_tokens": output_tokens,
                "requested_revision": self.revision,
                "resolved_model_commit": resolved if isinstance(resolved, str) else None,
                "actual_device": self._device,
                "actual_dtype": str(self._model.dtype) if self._model is not None else None,
                "weight_quantization": "nf4" if self.four_bit else None,
                "decoding": "pure_greedy", "repetition_penalty": 1.0,
                "input_truncated": False, "output_truncated": output_truncated,
                "weight_bytes_verified": False}

    def _render(self, request: Request) -> str:
        return self._tokenizer.apply_chat_template(
            [{"role": "system", "content": request.system}, {"role": "user", "content": request.prompt}],
            tokenize=False, add_generation_prompt=True)

    def _eos_ids(self) -> set:
        eos = getattr(self._model.generation_config, "eos_token_id", None)
        ids = set(eos if isinstance(eos, (tuple, list)) else [eos])
        ids.add(self._tokenizer.eos_token_id)
        ids.discard(None)
        return ids

    def _generate(self, input_ids, attention_mask, max_new_tokens: int, pad_token_id=None):
        torch = self._torch
        extra = {} if pad_token_id is None else {"pad_token_id": pad_token_id}
        self._started = time.monotonic()
        with torch.inference_mode():
            # Override the checkpoint's sampling defaults so decoding is plain argmax.
            return self._model.generate(input_ids=input_ids.to(self._device),
                                        attention_mask=attention_mask.to(self._device),
                                        max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                                        num_return_sequences=1, repetition_penalty=1.0, temperature=None,
                                        top_p=None, top_k=None, use_cache=True,
                                        return_dict_in_generate=False, **extra)

    def _finish(self, tokens: list, input_tokens: int, limit: int):
        """Cut one generated row at its first EOS; return (text or OutputTruncated, receipt)."""
        eos_ids = self._eos_ids()
        stop = next((i for i, token in enumerate(tokens) if token in eos_ids), None)
        used = tokens if stop is None else tokens[:stop + 1]
        truncated = stop is None and len(tokens) >= limit
        text = self._tokenizer.decode(used, skip_special_tokens=True)
        receipt = self._receipt(input_tokens, len(used), truncated)
        receipt["generation_seconds"] = round(time.monotonic() - getattr(self, "_started", time.monotonic()), 3)
        if truncated:
            return OutputTruncated("Generation reached its token limit without EOS; partial output "
                                   "kept for audit only.", text), receipt
        return text, receipt

    def complete(self, request: Request) -> str:
        self.last_receipt = None
        self._load_tokenizer()
        inputs = self._tokenizer(self._render(request), return_tensors="pt", add_special_tokens=False,
                                 truncation=False)
        length = inputs["input_ids"].shape[-1]
        self.last_receipt = self._receipt(length, 0)
        if length > self.max_input_tokens:
            raise GenerationFailure(f"Request contains {length} input tokens; limit is "
                                    f"{self.max_input_tokens}. No truncation applied.")
        self._load_model()
        self.last_receipt = self._receipt(length, 0)
        self._torch.manual_seed(request.seed)
        # Only what a causal LM consumes; some tokenizers also emit token_type_ids,
        # which generate() rejects for Qwen2-family models.
        mask = inputs["attention_mask"] if "attention_mask" in inputs else self._torch.ones_like(inputs["input_ids"])
        generated = self._generate(inputs["input_ids"], mask, request.max_new_tokens)
        outcome, self.last_receipt = self._finish(generated[0, length:].tolist(), length, request.max_new_tokens)
        if isinstance(outcome, OutputTruncated):
            raise outcome
        return outcome

    def complete_batch(self, requests: list[Request]) -> list:
        """Left-padded batched greedy generation: one reply text or OutputTruncated per request.

        Rows are cut at their own first EOS, so a row that finished early is not
        affected by the others' continued generation. Hard failures (input
        overflow, CUDA errors) raise for the whole batch. Padding can change
        floating-point rounding slightly compared with unbatched decoding; the
        result is still deterministic for the same batch.
        """
        self.last_receipts = None
        limits = {request.max_new_tokens for request in requests}
        if len(limits) != 1:
            results, receipts = [], []
            for request in requests:
                try:
                    results.append(self.complete(request))
                except OutputTruncated as exc:
                    results.append(exc)
                receipts.append(self.last_receipt)
            self.last_receipts = receipts
            return results
        limit = limits.pop()
        self._load_tokenizer()
        encoded = [self._tokenizer(self._render(r), add_special_tokens=False, truncation=False)["input_ids"]
                   for r in requests]
        lengths = [len(ids) for ids in encoded]
        if max(lengths) > self.max_input_tokens:
            self.last_receipts = [self._receipt(n, 0) for n in lengths]
            raise GenerationFailure(f"A batched request contains {max(lengths)} input tokens; limit is "
                                    f"{self.max_input_tokens}. No truncation applied.")
        self._load_model()
        torch = self._torch
        pad = self._tokenizer.pad_token_id
        pad = pad if pad is not None else next(iter(sorted(self._eos_ids())))
        width = max(lengths)
        input_ids = torch.tensor([[pad] * (width - n) + ids for ids, n in zip(encoded, lengths)])
        attention_mask = torch.tensor([[0] * (width - n) + [1] * n for n in lengths])
        generated = self._generate(input_ids, attention_mask, limit, pad_token_id=pad)
        results, receipts = [], []
        for row, n in zip(generated[:, width:].tolist(), lengths):
            outcome, receipt = self._finish(row, n, limit)
            results.append(outcome)
            receipts.append(receipt)
        self.last_receipts = receipts
        self.last_receipt = receipts[-1] if receipts else None
        return results
