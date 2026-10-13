"""Local persistence and an optional, lazily loaded Hugging Face backend."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from .core import (Config, LanguageModel, Observation, Perspective, Plan, Request,
                   RecursiveMirror, Result, Round, State, Task, canonical, digest)


CHECKPOINT_FORMAT = "recursive-mirror-checkpoint-v1"
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


def _string(value: object, label: str, limit: int = 12000) -> str:
    if type(value) is not str or not value.strip() or len(value) > limit:
        raise ValueError(f"{label} must be nonblank text of at most {limit} characters")
    return value


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
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


def state_from_dict(value: object) -> State:
    obj = _keys(value, {"strategy", "self_model", "version", "perspectives", "observations"})
    _string(obj["strategy"], "strategy")
    _string(obj["self_model"], "self_model")
    _integer(obj["version"], "version")
    perspectives = []
    for item in _array(obj["perspectives"], "perspectives"):
        row = _keys(item, {"round_index", "angle", "state_sha256", "text"})
        _integer(row["round_index"], "perspective round_index")
        if type(row["angle"]) is not str or row["angle"] not in RecursiveMirror.ANGLES:
            raise ValueError("Unknown perspective angle")
        _sha(row["state_sha256"], "perspective state_sha256")
        _string(row["text"], "perspective text")
        perspectives.append(Perspective(**row))
    observations = []
    phases = {"discovery", "validation_parent", "validation_candidate", "final_initial", "final_candidate"}
    for item in _array(obj["observations"], "observations"):
        row = _keys(item, {"round_index", "phase", "task_id", "category", "forecast_raw",
                          "probability", "answer", "correct", "prompt_sha256"})
        _integer(row["round_index"], "observation round_index", -1)
        if type(row["phase"]) is not str or row["phase"] not in phases:
            raise ValueError("Unknown observation phase")
        if (row["round_index"] == -1) != row["phase"].startswith("final_"):
            raise ValueError("Observation phase and round_index disagree")
        for key in ("task_id", "category", "forecast_raw", "answer"):
            _string(row[key], key)
        _sha(row["prompt_sha256"], "observation prompt_sha256")
        probability = row["probability"]
        if probability is not None and (type(probability) not in (int, float)
                                        or not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError("Observation probability must be null or a number in [0, 1]")
        raw = row["forecast_raw"].strip()
        expected_probability = int(raw) / 100 if re.fullmatch(r"(?:100|[1-9]?\d)", raw) else None
        if probability != expected_probability:
            raise ValueError("Observation probability disagrees with its raw forecast")
        if type(row["correct"]) is not bool:
            raise ValueError("Observation correct must be a boolean")
        observations.append(Observation(**row))
    return State(obj["strategy"], obj["self_model"], obj["version"],
                 tuple(perspectives), tuple(observations))


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
    if obj["format"] != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported checkpoint format")
    _string(obj["model_identity"], "model_identity")
    for field in ("plan_sha256", "state_sha256", "checkpoint_sha256"):
        _sha(obj[field], field)
    payload = {key: value for key, value in obj.items() if key != "checkpoint_sha256"}
    if digest(payload) != obj["checkpoint_sha256"]:
        raise ValueError("Checkpoint content hash mismatch")
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
        handle.write(canonical(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


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
            config: Config | None = None, initial: State | None = None) -> Result:
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    journal = Journal(directory / "journal.jsonl")
    try:
        _save_exclusive(directory / "plan.json", asdict(plan))
        engine = RecursiveMirror(model, plan, config, journal)
        result = engine.run(initial)
        checkpoint = checkpoint_for(result)
        _save_exclusive(directory / "result.json", asdict(result))
        _save_exclusive(directory / "checkpoint.json", checkpoint)
        journal({"event": "runtime_completed", "state_sha256": result.committed_state.sha256,
                 "checkpoint_sha256": checkpoint["checkpoint_sha256"]})
        return result
    except Exception as exc:
        journal({"event": "runtime_failed", "error_type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        journal.close()


class GenerationFailure(RuntimeError):
    def __init__(self, message: str, partial_output: str = ""):
        super().__init__(message)
        self.partial_output = partial_output


class HuggingFaceBackend:
    """Greedy inference. Importing this module never imports or downloads models.

    A 4096-token ceiling is a conservative guard for the initial v0.8 Colab
    canary, not a claim that larger requests are impossible. Partial generations
    are rejected and made available to the engine's error journal.
    """
    def __init__(self, model: str, *, allow_download: bool = False,
                 device: str = "auto", four_bit: bool = False,
                 revision: str | None = None):
        _string(model, "model")
        if revision is not None and (type(revision) is not str or not re.fullmatch(r"[0-9a-f]{40}", revision)):
            raise ValueError("revision must be a full 40-character lowercase hexadecimal commit")
        if allow_download and not Path(model).is_dir() and revision is None:
            raise ValueError("Remote downloads require an explicit immutable --revision commit")
        if device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be auto, cuda or cpu")
        if four_bit and device == "cpu":
            raise ValueError("Four-bit inference requires CUDA")
        self.model_name = model
        self.allow_download = allow_download
        self.device_request = device
        self.four_bit = four_bit
        self.revision = revision
        self.identity = (f"huggingface:{model}|revision={revision or 'unresolved-local-or-cache'}"
                         f"|device={device}|four_bit={str(four_bit).lower()}"
                         "|greedy|input_limit=4096|native_fp16_cuda")
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
        # T4 has native FP16, not native BF16. Keep CUDA arithmetic FP16.
        kwargs = {"local_files_only": not self.allow_download, "trust_remote_code": False,
                  "revision": self.revision,
                  "use_safetensors": True, "torch_dtype": torch.float16 if use_cuda else torch.float32}
        if self.four_bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16)
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
                "input_truncated": False, "output_truncated": output_truncated,
                "weight_bytes_verified": False}

    def complete(self, request: Request) -> str:
        self.last_receipt = None
        self._load_tokenizer()
        rendered = self._tokenizer.apply_chat_template(
            [{"role": "system", "content": request.system}, {"role": "user", "content": request.prompt}],
            tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(rendered, return_tensors="pt", add_special_tokens=False,
                                 truncation=False)
        length = inputs["input_ids"].shape[-1]
        self.last_receipt = self._receipt(length, 0)
        if length > 4096:
            raise GenerationFailure(f"Request contains {length} input tokens; limit is 4096. No truncation applied.")
        self._load_model()
        self.last_receipt = self._receipt(length, 0)
        torch = self._torch
        torch.manual_seed(request.seed)
        inputs = {name: value.to(self._device) for name, value in inputs.items()}
        with torch.inference_mode():
            generated = self._model.generate(**inputs, max_new_tokens=request.max_new_tokens,
                                             do_sample=False, num_beams=1, num_return_sequences=1,
                                             use_cache=True, return_dict_in_generate=False)
        tokens = generated[0, length:].tolist()
        output = self._tokenizer.decode(tokens, skip_special_tokens=True)
        eos = getattr(self._model.generation_config, "eos_token_id", None)
        eos_ids = set(eos if isinstance(eos, (tuple, list)) else [eos])
        eos_ids.add(self._tokenizer.eos_token_id)
        truncated = len(tokens) >= request.max_new_tokens and (not tokens or tokens[-1] not in eos_ids)
        self.last_receipt = self._receipt(length, len(tokens), truncated)
        if truncated:
            raise GenerationFailure("Generation reached its token limit without EOS; partial output rejected.", output)
        return output
