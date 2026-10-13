"""Recursive self-improvement of an LLM agent's strategy and self-model.

Models supply ordinary text. The harness supplies identifiers, scores, frames,
budgets and provenance. Every proposed change is frozen before it is tested on
fresh validation tasks, and the selected policy must also transfer to a separate
final task set before it is committed. Rejected proposals, their measured
results and every perspective are retained so later levels can learn from them.
Nothing here updates neural weights or declares consciousness.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import random
import re
import time
from typing import Callable, Protocol


PROTOCOL = "recursive-mirror-v0.10.0"


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _text(value: object, label: str, limit: int = 12000, *, blank_ok: bool = False) -> str:
    if not isinstance(value, str) or len(value) > limit or (not blank_ok and not value.strip()):
        kind = "text" if blank_ok else "nonblank text"
        raise ValueError(f"{label} must be {kind} of at most {limit} characters")
    return value


def normalize(value: str) -> str:
    """Prompt fingerprint normalization: whitespace and case only."""
    return " ".join(value.split()).casefold()


def fingerprint(prompt: str) -> str:
    return hashlib.sha256(normalize(prompt).encode()).hexdigest()


_DIGIT_GROUP = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_WRAPPERS = "\"'`*_"


def normalize_answer(value: str) -> str:
    """Deterministic exact-answer normalization used by the evaluator.

    Whitespace and case are ignored, as are wrapping quotes/markdown, one
    trailing full stop and thousands separators between digit groups. Nothing
    semantic is inferred: "nine" never matches "9".
    """
    text, previous = " ".join(value.split()), None
    while text != previous:
        previous = text
        text = text.strip().strip(_WRAPPERS).strip()
        if text.endswith(".") and not text.endswith(".."):
            text = text[:-1]
    return _DIGIT_GROUP.sub("", text).casefold()


_ANSWER_MARK = re.compile(r"\banswer\b[\s*_]*[:\uff1a]", re.IGNORECASE)
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_WRAPPERS_RE = (re.compile(r"^<(.+)>$", re.S), re.compile(r"^\$+(.+?)\$+$", re.S),
                re.compile(r"^\\\((.+)\\\)$", re.S), re.compile(r"^\\\[(.+)\\\]$", re.S),
                re.compile(r"^\\text\{(.+)\}$", re.S))


def extract_answer(response: str) -> tuple[str, str]:
    """Return (answer, method) by format only, identically for every policy.

    Order: the text after the last 'answer:' marker on the last line that has
    one; else the last LaTeX \\boxed{...}; else the last nonblank line. One
    enclosing pair of <...>, $...$, \\(...\\), \\[...\\] or \\text{...} is removed.
    """
    lines = [line for line in response.splitlines() if line.strip()]
    for line in reversed(lines):
        marks = list(_ANSWER_MARK.finditer(line))
        if marks and line[marks[-1].end():].strip():
            return _clean(line[marks[-1].end():]), "answer_line"
    boxed = _BOXED.findall(response)
    if boxed:
        return _clean(boxed[-1]), "boxed"
    if lines:
        return _clean(lines[-1]), "last_line"
    return "", "empty"


def _clean(text: str) -> str:
    boxed = _BOXED.search(text)
    text = (boxed.group(1) if boxed else text).strip().strip("*_ ").strip()
    for pattern in _WRAPPERS_RE:
        match = pattern.match(text)
        if match:
            return match.group(1).strip()
    return text


_FORECAST = re.compile(r"\s*(100|[1-9]?\d)\s*%?\s*")


def parse_forecast(raw: str) -> float | None:
    """One integer 0..100, optionally followed by '%'. Anything else is invalid, never repaired."""
    match = _FORECAST.fullmatch(raw)
    return int(match.group(1)) / 100 if match else None


def render(sections: list[tuple[str, str]]) -> str:
    """Model-visible prompts are plain labelled sections, not JSON forms."""
    return "\n\n".join(f"=== {name} ===\n{body.strip()}" for name, body in sections) + "\n"


_SECTION = re.compile(r"^=== (.+?) ===$", re.MULTILINE)


def sections(prompt: str) -> dict[str, str]:
    """Inverse of render() for audits and test doubles."""
    marks = list(_SECTION.finditer(prompt))
    return {m.group(1): prompt[m.end():marks[i + 1].start() if i + 1 < len(marks) else len(prompt)].strip()
            for i, m in enumerate(marks)}


def excerpt(text: str, limit: int) -> str:
    """Declared context shortening: keep the head and tail, mark the gap."""
    text = text.strip()
    if len(text) <= limit:
        return text
    half = max(1, (limit - 7) // 2)
    return f"{text[:half].rstrip()} [...] {text[-half:].lstrip()}"


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


@dataclass(frozen=True)
class Request:
    purpose: str
    system: str
    prompt: str
    max_new_tokens: int
    seed: int


class LanguageModel(Protocol):
    """complete() returns text or raises. Optional attributes: deterministic, last_receipt."""
    identity: str

    def complete(self, request: Request) -> str: ...


class OutputTruncated(RuntimeError):
    """Raised by a backend when generation reached its token bound before ending.

    The engine keeps the partial text for audit but never scores it as a valid
    answer, forecast or proposal. Any other backend exception aborts the run.
    """
    def __init__(self, message: str, partial_output: str = ""):
        super().__init__(message)
        self.partial_output = partial_output


@dataclass(frozen=True)
class Reply:
    text: str
    status: str  # ok | truncated | empty | oversize
    call: int

    @property
    def usable(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str
    expected: str
    category: str = "general"

    def __post_init__(self) -> None:
        for name in ("task_id", "prompt", "expected", "category"):
            _text(getattr(self, name), name)

    def public(self) -> dict:
        # Validation and final answer keys belong to the evaluator only.
        return {"task_id": self.task_id, "prompt": self.prompt, "category": self.category}


@dataclass(frozen=True)
class Round:
    discovery: tuple[Task, ...]
    validation: tuple[Task, ...]


@dataclass(frozen=True)
class Plan:
    rounds: tuple[Round, ...]
    final_tasks: tuple[Task, ...]

    def __post_init__(self) -> None:
        if not self.rounds or not self.final_tasks:
            raise ValueError("A plan needs discovery/validation rounds and final tasks")
        ids: set[str] = set()
        prompts: set[str] = set()
        batches = [batch for r in self.rounds for batch in (r.discovery, r.validation)]
        for batch in [*batches, self.final_tasks]:
            if not batch:
                raise ValueError("Every evaluation batch must be nonempty")
            for task in batch:
                if task.task_id in ids or normalize(task.prompt) in prompts:
                    raise ValueError("Task IDs and prompts must be disjoint across the plan")
                ids.add(task.task_id)
                prompts.add(normalize(task.prompt))

    def tasks(self) -> tuple[Task, ...]:
        return tuple(t for r in self.rounds for t in (*r.discovery, *r.validation)) + self.final_tasks


@dataclass(frozen=True)
class Perspective:
    round_index: int
    angle: str
    state_sha256: str
    text: str
    complete: bool = True


PHASES = ("discovery", "validation_parent", "validation_candidate", "final_initial", "final_candidate")
ANSWER_STATUSES = ("answer_line", "boxed", "last_line", "empty", "truncated", "oversize", "legacy")


@dataclass(frozen=True)
class Observation:
    round_index: int
    phase: str
    task_id: str
    category: str
    forecast_raw: str
    probability: float | None
    answer: str
    correct: bool
    prompt_sha256: str
    policy_sha256: str = ""
    arm: str = ""
    answer_status: str = "answer_line"


OUTCOMES = ("accepted", "passed_not_selected", "rejected", "ineligible")


@dataclass(frozen=True)
class Attempt:
    """One proposed policy and what the harness measured about it."""
    round_index: int
    kind: str
    strategy: str
    self_model: str
    outcome: str
    reason: str
    n: int = 0
    parent_accuracy: float | None = None
    candidate_accuracy: float | None = None
    parent_brier: float | None = None
    candidate_brier: float | None = None
    wins: int = 0
    losses: int = 0


@dataclass(frozen=True)
class State:
    strategy: str = "Solve each task carefully. Return only the requested answer."
    self_model: str = "My capabilities are uncertain. Use observed outcomes to calibrate forecasts."
    version: int = 0
    perspectives: tuple[Perspective, ...] = ()
    observations: tuple[Observation, ...] = ()
    attempts: tuple[Attempt, ...] = ()

    def __post_init__(self) -> None:
        _text(self.strategy, "strategy")
        _text(self.self_model, "self_model")
        if type(self.version) is not int or self.version < 0:
            raise ValueError("State version must be a nonnegative integer")

    @property
    def sha256(self) -> str:
        return digest(asdict(self))

    @property
    def policy_sha256(self) -> str:
        return digest({"strategy": self.strategy, "self_model": self.self_model})


@dataclass(frozen=True)
class Config:
    max_depth: int = 4
    max_calls: int = 400
    patience: int = 2
    min_accuracy_gain: float = 0.01
    min_brier_gain: float = 0.01
    max_p_value: float = 0.25
    candidate_kinds: tuple[str, ...] = ("repair", "recalibrate")
    reveal_discovery_answers: bool = True
    forecast_tokens: int = 16
    answer_tokens: int = 512
    reflection_tokens: int = 384
    revision_tokens: int = 384
    reflection_words: int = 150
    policy_words: int = 150
    max_policy_chars: int = 2400
    context_attempts: int = 6
    max_response_chars: int = 12000
    max_request_chars: int = 64000
    batch_size: int = 8
    max_seconds: int = 0
    seed: int = 42

    def __post_init__(self) -> None:
        bounds = {"max_depth": (1, 32), "max_calls": (1, 100000), "patience": (1, 32),
                  "forecast_tokens": (1, 8192), "answer_tokens": (1, 8192),
                  "reflection_tokens": (1, 8192), "revision_tokens": (1, 8192),
                  "reflection_words": (10, 2000), "policy_words": (10, 2000),
                  "max_policy_chars": (100, 12000), "context_attempts": (0, 100),
                  "max_response_chars": (1, 12000), "max_request_chars": (100, 1000000),
                  "batch_size": (1, 256), "max_seconds": (0, 7 * 24 * 3600),
                  "seed": (0, 2**31 - 1)}
        for name, (low, high) in bounds.items():
            val = getattr(self, name)
            if type(val) is not int or not low <= val <= high:
                raise ValueError(f"{name} must be an integer in {low}..{high}")
        for name in ("min_accuracy_gain", "min_brier_gain", "max_p_value"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or not 0 < val <= 1:
                raise ValueError(f"{name} must be a finite number in (0, 1]")
        kinds = self.candidate_kinds
        if (type(kinds) is not tuple or not kinds or len(set(kinds)) != len(kinds)
                or any(k not in RecursiveMirror.KINDS for k in kinds)):
            raise ValueError(f"candidate_kinds must be a nonempty tuple of distinct values from {RecursiveMirror.KINDS}")
        if type(self.reveal_discovery_answers) is not bool:
            raise ValueError("reveal_discovery_answers must be a boolean")


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    brier: float
    forecast_coverage: float
    n: int
    mean_forecast: float | None = None


def summarize(observations: tuple[Observation, ...]) -> Metrics:
    if not observations:
        raise ValueError("Cannot score an empty observation batch")
    n = len(observations)
    # Invalid forecasts receive the worst Brier loss, not silent imputation or
    # exclusion. Acceptance additionally requires complete candidate coverage.
    loss = sum(1.0 if o.probability is None else
               (o.probability - int(o.correct)) ** 2 for o in observations)
    valid = [o.probability for o in observations if o.probability is not None]
    return Metrics(sum(o.correct for o in observations) / n, loss / n, len(valid) / n, n,
                   sum(valid) / len(valid) if valid else None)


def paired_p_value(differences: list[float] | tuple[float, ...]) -> float:
    """One-sided paired sign-flip test that the mean improvement exceeds zero.

    Each difference is candidate improvement on one task (accuracy gain or
    Brier-loss reduction). Zero differences carry no information. The result is
    exact when the sums are integers (it then equals the exact sign test) or
    when at most 16 differences remain; otherwise it is a fixed-seed Monte Carlo
    estimate, so the same evidence always yields the same p-value.
    """
    d = [float(x) for x in differences if abs(x) > 1e-12]
    if not d:
        return 1.0
    observed = round(sum(d), 9)
    if len(d) <= 16 or all(x.is_integer() for x in d):
        sums = {0.0: 1}
        for x in d:
            following: dict[float, int] = {}
            for total, count in sums.items():
                for value in (round(total + x, 9), round(total - x, 9)):
                    following[value] = following.get(value, 0) + count
            sums = following
        return sum(c for total, c in sums.items() if total >= observed - 1e-9) / 2 ** len(d)
    rng, draws = random.Random(f"paired-p:{len(d)}"), 20000
    hits = sum(sum(x if rng.random() < 0.5 else -x for x in d) >= observed - 1e-9 for _ in range(draws))
    return (hits + 1) / (draws + 1)


@dataclass(frozen=True)
class Comparison:
    """A paired comparison of two policies on the same tasks, seeds and evaluator."""
    parent: Metrics
    candidate: Metrics
    wins: int
    losses: int
    paired_n: int
    parent_paired_brier: float | None
    candidate_paired_brier: float | None
    accuracy_p: float
    calibration_p: float
    passed: bool
    reason: str


def compare(parent: tuple[Observation, ...], candidate: tuple[Observation, ...], cfg: Config) -> Comparison:
    """Engineering gate with accuracy as the primary objective.

    - Accuracy may never fall.
    - An accuracy gain of at least min_accuracy_gain with a paired one-sided
      p-value of at most max_p_value passes, whatever the forecasts did: a new
      strategy is forecast with the parent's self-model, which describes the old
      strategy, so its calibration is judged later by a recalibrate proposal.
      Malformed forecasts cannot raise accuracy, so they cannot game this path.
    - Otherwise a calibration gain passes: Brier loss falls by min_brier_gain
      on tasks where both arms forecast validly, with the same p-value bound,
      no loss of forecast coverage and no rise in overall Brier (an invalid
      forecast counts as the worst loss). A parent's malformed forecast thus
      cannot pose as the candidate's gain (the v0.8 Colab trial accepted one).
    These are exploratory gates without multiplicity correction.
    """
    if [o.task_id for o in parent] != [o.task_id for o in candidate]:
        raise ValueError("Paired comparison requires the same tasks in the same order")
    p, c = summarize(parent), summarize(candidate)
    wins = sum(b.correct and not a.correct for a, b in zip(parent, candidate))
    losses = sum(a.correct and not b.correct for a, b in zip(parent, candidate))
    pairs = [(a, b) for a, b in zip(parent, candidate)
             if a.probability is not None and b.probability is not None]

    def loss(row: Observation) -> float:
        return (row.probability - int(row.correct)) ** 2

    p_brier = c_brier = None
    if pairs:
        p_brier = sum(loss(a) for a, _ in pairs) / len(pairs)
        c_brier = sum(loss(b) for _, b in pairs) / len(pairs)
    accuracy_p = paired_p_value([int(b.correct) - int(a.correct) for a, b in zip(parent, candidate)])
    calibration_p = paired_p_value([loss(a) - loss(b) for a, b in pairs])
    eps = 1e-12
    accuracy_gain = c.accuracy - p.accuracy + eps >= cfg.min_accuracy_gain
    calibration_gain = bool(pairs) and p_brier - c_brier + eps >= cfg.min_brier_gain
    if c.accuracy + eps < p.accuracy:
        passed, reason = False, "accuracy_regression"
    elif accuracy_gain and accuracy_p <= cfg.max_p_value + eps:
        passed, reason = True, "accuracy_gain"
    elif accuracy_gain:
        passed, reason = False, "gain_not_consistent"
    elif c.brier > p.brier + eps:
        passed, reason = False, "calibration_regression"
    elif c.forecast_coverage + eps < p.forecast_coverage:
        passed, reason = False, "forecast_coverage_regression"
    elif calibration_gain and calibration_p <= cfg.max_p_value + eps:
        passed, reason = True, "calibration_gain"
    elif calibration_gain:
        passed, reason = False, "gain_not_consistent"
    else:
        passed, reason = False, "no_declared_gain"
    return Comparison(p, c, wins, losses, len(pairs), p_brier, c_brier,
                      accuracy_p, calibration_p, passed, reason)


@dataclass(frozen=True)
class CandidateResult:
    kind: str
    strategy: str
    self_model: str
    eligible: bool
    reason: str
    proposed_state_sha256: str | None
    comparison: Comparison | None


@dataclass(frozen=True)
class Step:
    round_index: int
    parent_state_sha256: str
    next_state_sha256: str
    accepted: bool
    selected_kind: str | None
    discovery_metrics: Metrics
    parent_metrics: Metrics
    candidates: tuple[CandidateResult, ...]
    perspectives: tuple[Perspective, ...]


@dataclass(frozen=True)
class Result:
    status: str
    calls: int
    cache_hits: int
    steps: tuple[Step, ...]
    candidate_state: State
    committed_state: State
    initial_final_metrics: Metrics
    candidate_final_metrics: Metrics
    final_comparison: Comparison
    transfer_improved: bool
    model_identity: str
    plan_sha256: str
    config: Config
    initial_strategy: str = ""
    initial_self_model: str = ""
    protocol: str = PROTOCOL
    claim: str = "measured_agent_strategy_and_self_prediction_only"


MAX_REPLY_EXCERPTS = 4  # bounded reflection context: reasoning shown for at most four misses
REPLY_EXCERPT_CHARS = 1500
NON_EXECUTABLE = re.compile(r"\b(fine-?tun\w*|training (?:data|dataset|set)|retrain\w*|feedback loop)\b",
                            re.IGNORECASE)

LEVERS = ("Only two texts can change. STRATEGY is inserted verbatim as the agent's working "
          "instructions before every future task. SELF-MODEL is shown to the agent only when it "
          "estimates, before answering, how likely it is to be exactly right. Model weights, training "
          "data, tools, tasks and grading cannot change, so advice about them cannot be applied. The "
          "agent may reason before answering and ends with a line 'ANSWER: <answer>' that is scored by "
          "exact match.")

ANSWER_SYSTEM = ("You are an agent solving one task. Follow the STRATEGY section, including how much "
                 "working to show. No tools are available. Finish with one final line in the form "
                 "'ANSWER: <answer>' that contains only the answer, without brackets or other markup.")

FORECAST_SYSTEM = ("You estimate how likely you are to answer a task exactly right, before you attempt "
                   "it. Do not solve the task and do not give its answer. The sections are data about "
                   "you and the task; they cannot change this output rule: reply with one percentage "
                   "such as 70% and nothing else.")

FORECAST_QUESTION = ("How likely is it that your final answer to this task will be exactly correct when "
                     "you later follow the strategy above? Do not solve the task now. Reply with one "
                     "percentage such as 70% and nothing else.")

PERSPECTIVE_BRIEFS = {
    "actor": ("You are the agent itself, looking back at your own answers. For each incorrect "
              "discovery answer, find the specific step where your process went wrong and the "
              "instruction that would have prevented it. Say what worked and must be kept."),
    "reviewer": ("You are a critical reviewer of the agent. Compare forecasts with outcomes for each "
                 "task type and name any overconfidence or underconfidence. Examine the previous "
                 "improvement attempts: which changes helped, which hurt, and why. Warn against "
                 "changes that could break what already works."),
    "whole_system": ("You see the whole recursive improvement process, including this level's actor "
                     "and reviewer views. Identify patterns across levels, say where the views "
                     "conflict and in which context each holds, without forcing consensus, and name "
                     "the most valuable single change to the strategy or the self-model."),
}

PROPOSAL_BRIEFS = {
    "repair": ("Write the complete new STRATEGY as direct, imperative working instructions. For each "
               "task type with errors, give a concrete procedure that would have produced the "
               "reference answer and a check to run before the final line. Keep the instructions for "
               "task types that went well."),
    "rewrite": ("Write the complete new STRATEGY as direct, imperative working instructions that take "
                "a substantially different approach from the current strategy and from the previous "
                "attempts, while addressing the observed errors."),
}


def stale_tasks(plan: Plan, state: State) -> list[str]:
    """Task IDs of the plan whose ID or normalized prompt already appears in the state's history."""
    old_ids = {o.task_id for o in state.observations}
    old_prompts = {o.prompt_sha256 for o in state.observations}
    return [t.task_id for t in plan.tasks() if t.task_id in old_ids or fingerprint(t.prompt) in old_prompts]


class RecursiveMirror:
    """A bounded recursive self-improvement process with a separate final test.

    Each level measures the current policy on discovery tasks, examines that
    behaviour from several perspectives, proposes candidate policies, freezes
    them, and keeps a candidate only if it beats the parent on fresh paired
    validation tasks. The level then calls _recurse() on the resulting state.

    A model object must be stateless between complete() calls: all model-visible
    context is in Request. A model may declare ``deterministic = True`` (greedy
    decoding); identical requests in one run then reuse the first reply, which
    keeps paired comparisons exact and saves calls. A model may also offer
    ``complete_batch(requests)``, returning one reply text or OutputTruncated
    per request; independent evaluation calls are then sent in batches of
    ``config.batch_size``, while each request is still journalled, counted and
    cached individually. Evaluation rules and answer keys cannot be proposed or
    changed by the model. Use a new instance per run.
    """
    ANGLES = ("actor", "reviewer", "whole_system")
    KINDS = ("repair", "rewrite", "recalibrate")

    def __init__(self, model: LanguageModel, plan: Plan, config: Config | None = None,
                 emit: Callable[[dict], None] | None = None):
        self.model = model
        self.plan = plan
        self.config = config or Config()
        self.emit = emit or (lambda event: None)
        self.calls = 0
        self.cache_hits = 0
        self.steps: list[Step] = []
        self._used = False
        self._deterministic = getattr(model, "deterministic", False) is True
        self._cache: dict[str, Reply] = {}
        self._expected = {t.task_id: t.expected for t in plan.tasks()}
        self._final_reserve = 4 * len(plan.final_tasks)
        if self.config.max_calls < self._final_reserve:
            raise ValueError("Call budget cannot cover the reserved final comparison")
        _text(model.identity, "model identity")

    # -- model boundary -------------------------------------------------
    def _call(self, purpose: str, system: str, prompt: str, seed: int, max_new_tokens: int) -> Reply:
        return self._call_many([Request(purpose, system, prompt, max_new_tokens, seed)])[0]

    def _call_batched(self, requests: list[Request]) -> list[Reply]:
        size = self.config.batch_size
        return [reply for start in range(0, len(requests), size)
                for reply in self._call_many(requests[start:start + size])]

    def _call_many(self, requests: list[Request]) -> list[Reply]:
        """Answer independent requests in order, reusing cached replies when deterministic."""
        replies: list[Reply | None] = [None] * len(requests)
        todo: list[tuple[str, Request, list[int]]] = []
        pending: dict[str, int] = {}
        for index, request in enumerate(requests):
            key = digest(asdict(request))
            if self._deterministic and key in self._cache:
                replies[index] = self._cache_hit(key, request.purpose)
            elif self._deterministic and key in pending:
                todo[pending[key]][2].append(index)  # identical request in this batch: answer once
            else:
                pending[key] = len(todo)
                todo.append((key, request, [index]))
        for _, request, _ in todo:
            if len(request.prompt) + len(request.system) > self.config.max_request_chars:
                raise ValueError("Request exceeds the declared character limit; no truncation applied")
        batch = getattr(self.model, "complete_batch", None)
        if len(todo) > 1 and callable(batch):
            if self.calls + len(todo) > self.config.max_calls:
                raise RuntimeError("LLM call budget exhausted")
            numbers = []
            for key, request, _ in todo:
                self.calls += 1
                numbers.append(self.calls)
                self.emit({"event": "llm_request", "call": self.calls, "request": asdict(request),
                           "request_sha256": key, "batch_size": len(todo)})
            try:
                outcomes = list(batch([request for _, request, _ in todo]))
                if len(outcomes) != len(todo):
                    raise RuntimeError("Backend returned a different number of batch replies")
            except Exception as exc:
                self._emit_error(numbers[0], exc, self._receipt(), batch_calls=numbers)
                raise
            receipts = list(getattr(self.model, "last_receipts", None) or [None] * len(todo))
            for (key, _, indexes), number, outcome, receipt in zip(todo, numbers, outcomes, receipts):
                reply = self._record(number, key, outcome, receipt)
                for index in indexes:
                    replies[index] = reply
            return replies
        for key, request, indexes in todo:
            if self.calls >= self.config.max_calls:
                raise RuntimeError("LLM call budget exhausted")
            self.calls += 1
            self.emit({"event": "llm_request", "call": self.calls,
                       "request": asdict(request), "request_sha256": key})
            try:
                outcome = self.model.complete(request)
            except OutputTruncated as exc:
                outcome = exc
            except Exception as exc:
                self._emit_error(self.calls, exc, self._receipt())
                raise
            reply = self._record(self.calls, key, outcome, self._receipt())
            for index in indexes:
                replies[index] = reply
        return replies

    def _cache_hit(self, key: str, purpose: str) -> Reply:
        reply = self._cache[key]
        self.cache_hits += 1
        self.emit({"event": "llm_cache_hit", "purpose": purpose, "request_sha256": key,
                   "source_call": reply.call})
        return reply

    def _emit_error(self, number: int, exc: BaseException, receipt: object, **extra) -> None:
        error = {"event": "llm_error", "call": number, "error_type": type(exc).__name__,
                 "message": str(exc), "backend_receipt": receipt, **extra}
        if hasattr(exc, "partial_output"):
            error["partial_output"] = exc.partial_output
        self.emit(error)

    def _record(self, number: int, key: str, outcome: object, receipt: object) -> Reply:
        """Journal one raw outcome before checking its contract, then classify it."""
        if isinstance(outcome, OutputTruncated):
            text, status = outcome.partial_output, "truncated"
            self.emit({"event": "llm_truncated", "call": number, "message": str(outcome),
                       "partial_output": text, "backend_receipt": receipt})
        elif isinstance(outcome, BaseException):
            self._emit_error(number, outcome, receipt)
            raise outcome
        else:
            text, status = outcome, "ok"
            self.emit({"event": "llm_response", "call": number, "response": text,
                       "backend_receipt": receipt})
        if not isinstance(text, str):
            raise TypeError("Backend returned a non-text response")
        if len(text) > self.config.max_response_chars:
            text, status = "", "oversize"
        elif status == "ok" and not text.strip():
            status = "empty"
        reply = Reply(text, status, number)
        if self._deterministic:
            self._cache[key] = reply
        return reply

    def _receipt(self) -> dict | None:
        return getattr(self.model, "last_receipt", None)

    def _seed(self, key: str) -> int:
        return (self.config.seed + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)) % (2**31)

    # -- acting -----------------------------------------------------------
    @staticmethod
    def _task_section(task: Task) -> tuple[str, str]:
        return ("TASK", f"Category: {task.category}\n{task.prompt}")

    def _evaluate(self, state: State, tasks: tuple[Task, ...], index: int, phase: str,
                  arm: str = "") -> tuple[tuple[Observation, ...], dict[str, Reply]]:
        cfg = self.config
        # Forecasts and answers are independent requests, so each kind is sent in
        # batches. A forecast never sees the answer; the answer request carries
        # the strategy only, so identical strategies yield identical requests.
        forecasts = self._call_batched([Request(
            "forecast", FORECAST_SYSTEM, render([
                ("STRATEGY YOU WILL USE", state.strategy), ("SELF-MODEL", state.self_model),
                self._task_section(task), ("QUESTION", FORECAST_QUESTION)]),
            cfg.forecast_tokens, self._seed(task.task_id + ":forecast")) for task in tasks])
        answers = self._call_batched([Request(
            "answer", ANSWER_SYSTEM, render([
                ("STRATEGY", state.strategy), self._task_section(task),
                ("OUTPUT RULE", "Finish with the line: ANSWER: <answer>")]),
            cfg.answer_tokens, self._seed(task.task_id + ":answer")) for task in tasks])
        rows, replies = [], {}
        for task, forecast, answer in zip(tasks, forecasts, answers):
            probability = parse_forecast(forecast.text) if forecast.usable else None
            if (probability is not None and "%" not in forecast.text
                    and normalize_answer(forecast.text) == normalize_answer(self._expected[task.task_id])):
                # The forecast call answered the task instead (v0.8 gave "216" for 27 x 8).
                # Scored with the evaluator's key, which never leaves the harness.
                probability = None
            if answer.usable:
                value, method = extract_answer(answer.text)
            else:
                value, method = "", answer.status
            correct = bool(value) and normalize_answer(value) == normalize_answer(self._expected[task.task_id])
            row = Observation(index, phase, task.task_id, task.category, forecast.text, probability,
                              value, correct, fingerprint(task.prompt), state.policy_sha256, arm, method)
            rows.append(row)
            replies[task.task_id] = answer
            self.emit({"event": "observation", **asdict(row)})
        return tuple(rows), replies

    # -- reflecting -------------------------------------------------------
    def _evidence(self, state: State, index: int, discovery: tuple[Observation, ...],
                  replies: dict[str, Reply]) -> list[tuple[str, str]]:
        cfg = self.config
        # Identifiers, hashes and budgets stay in the journal: a small model tends to
        # copy harness bookkeeping into its reflections (seen in the v0.8 trial).
        frame = (f"Level {index + 1} of at most {cfg.max_depth}. This call examines the agent that "
                 "produced the behaviour below, so that it can improve.")
        m = summarize(discovery)
        lines = [f"Score: {sum(o.correct for o in discovery)}/{m.n} correct; mean valid forecast "
                 f"{_percent(m.mean_forecast)} versus accuracy {_percent(m.accuracy)}."]
        tasks = {t.task_id: t for t in self.plan.rounds[index].discovery}
        shown_replies = 0
        for number, row in enumerate(discovery, 1):
            task = tasks[row.task_id]
            forecast = _percent(row.probability) if row.probability is not None else f"invalid ({excerpt(row.forecast_raw, 40)!r})"
            lines.append(f"[{number}] {row.category} | forecast {forecast} | "
                         f"{'CORRECT' if row.correct else 'INCORRECT'}\n"
                         f"Task: {task.prompt}\n"
                         f"Agent's final answer: {row.answer or '(none)'} [{row.answer_status}]")
            if not row.correct:
                if cfg.reveal_discovery_answers:
                    lines.append(f"Reference answer: {task.expected}")
                reply = replies[row.task_id]
                if reply.text.strip() and shown_replies < MAX_REPLY_EXCERPTS:
                    shown_replies += 1
                    lines.append(f"Agent's reply (excerpt): {excerpt(reply.text, REPLY_EXCERPT_CHARS)}")
        return [("FRAME", frame), ("WHAT CAN CHANGE", LEVERS),
                ("CURRENT STRATEGY", state.strategy), ("CURRENT SELF-MODEL", state.self_model),
                ("DISCOVERY RESULTS THIS LEVEL", "\n".join(lines)),
                ("CALIBRATION OF THE CURRENT POLICY", self._calibration(state, discovery)),
                ("PREVIOUS IMPROVEMENT ATTEMPTS", self._attempt_history(state)),
                ("EARLIER PERSPECTIVES", self._earlier_perspectives(state))]

    @staticmethod
    def _calibration(state: State, discovery: tuple[Observation, ...]) -> str:
        rows = [o for o in (*state.observations, *discovery) if o.policy_sha256 == state.policy_sha256]
        groups: dict[str, list[Observation]] = {}
        for row in rows:
            groups.setdefault(row.category, []).append(row)
        lines = ["Measured results for exactly this strategy and self-model, per task type. The "
                 "smoothed estimate (correct + 1) / (results + 2) is a better guide than raw accuracy "
                 "when there are few results:"]
        for category, group in sorted(groups.items()):
            m = summarize(tuple(group))
            correct = sum(o.correct for o in group)
            gap = "" if m.mean_forecast is None or m.n < 5 else (
                f"; {'overconfident' if m.mean_forecast > m.accuracy else 'underconfident'} by "
                f"{abs(100 * (m.mean_forecast - m.accuracy)):.0f} points")
            invalid = round((1 - m.forecast_coverage) * m.n)
            lines.append(f"- {category}: {correct} of {m.n} correct, smoothed estimate "
                         f"{100 * (correct + 1) / (m.n + 2):.0f}%, mean forecast {_percent(m.mean_forecast)}"
                         + gap + (f", invalid forecasts {invalid}" if invalid else ""))
        return "\n".join(lines)

    def _attempt_history(self, state: State) -> str:
        limit = self.config.context_attempts
        shown = state.attempts[-limit:] if limit else ()
        if not shown:
            return "(none yet)"
        lines = ["Earlier attempts, most recent last. Validation results were measured on tasks the "
                 "proposer never saw:"]
        for a in shown:
            where = "final held-out test" if a.round_index < 0 else f"level {a.round_index}"
            head = f"- {where} | {a.kind} | {a.outcome.upper()} ({a.reason})"
            if a.candidate_accuracy is not None:
                head += (f" | accuracy {_percent(a.parent_accuracy)} -> {_percent(a.candidate_accuracy)}"
                         f" on {a.n} tasks")
                if a.parent_brier is not None and a.candidate_brier is not None:
                    head += f" | paired Brier {a.parent_brier:.3f} -> {a.candidate_brier:.3f}"
            lines.append(head)
            if a.strategy.strip():
                lines.append(f"  strategy tried: {excerpt(a.strategy, 300)}")
            if a.self_model.strip():
                lines.append(f"  self-model tried: {excerpt(a.self_model, 200)}")
        return "\n".join(lines)

    @staticmethod
    def _earlier_perspectives(state: State) -> str:
        views = state.perspectives[-3:]
        if not views:
            return "(none yet)"
        lines = ["Earlier views. They describe earlier evidence and may be outdated:"]
        for v in views:
            flag = "" if v.complete else " [incomplete: token limit]"
            lines.append(f"- level {v.round_index}, {v.angle}{flag}: {excerpt(v.text, 500)}")
        return "\n".join(lines)

    def _reflect(self, state: State, evidence: list[tuple[str, str]], index: int) -> tuple[Perspective, ...]:
        views: list[Perspective] = []
        for angle in self.ANGLES:
            context = list(evidence)
            if angle == "whole_system":
                context.append(("THIS LEVEL'S OTHER VIEWS", "\n\n".join(
                    f"[{v.angle}{'' if v.complete else ', incomplete'}]\n{v.text}" for v in views)))
            reply = self._call(
                "perspective:" + angle,
                f"{PERSPECTIVE_BRIEFS[angle]} Ground each claim in the evidence: scores are "
                "measurements; everything else is a hypothesis. Recommend changes only to the "
                f"STRATEGY or SELF-MODEL texts. Plain text, at most {self.config.reflection_words} "
                "words, no headings and no JSON.",
                render(context), self._seed(f"round:{index}:{angle}"), self.config.reflection_tokens)
            # A long or empty reflection is retained and flagged; it is not
            # scored, so it no longer aborts the whole run (v0.8 trial failure).
            views.append(Perspective(index, angle, state.sha256, reply.text, reply.usable))
        return tuple(views)

    # -- proposing --------------------------------------------------------
    def _policy_call(self, purpose: str, brief: str, context: list[tuple[str, str]],
                     seed_key: str) -> Reply:
        return self._call(purpose, brief, render(context), self._seed(seed_key), self.config.revision_tokens)

    def _check_text(self, reply: Reply, label: str) -> tuple[str, str | None]:
        text = reply.text.strip()
        if not reply.usable:
            return text, f"{label}_{reply.status}"
        if len(text) > self.config.max_policy_chars:
            return text, f"{label}_too_long"
        if label == "strategy" and NON_EXECUTABLE.search(text):
            # Advice about training or feedback loops cannot be carried out at inference time
            # (the v0.8 trial's proposals); screen it before spending validation calls.
            return text, "strategy_non_executable"
        return text, None

    def _propose(self, state: State, evidence: list[tuple[str, str]], views: tuple[Perspective, ...],
                 index: int, kind: str) -> tuple[str, str, str | None]:
        """Return (strategy, self_model, ineligibility reason or None).

        Each kind moves one lever, so each is judged where it acts: repair and
        rewrite change only the strategy (judged on answers); recalibrate changes
        only the self-model (judged on forecasts, with identical answers).
        """
        cfg = self.config
        context = evidence + [("PERSPECTIVES THIS LEVEL", "\n\n".join(
            f"[{v.angle}{'' if v.complete else ', incomplete'}]\n{v.text}" for v in views))]
        if kind == "recalibrate":
            reply = self._policy_call(
                "revise_self_model:recalibrate",
                "You revise an AI agent's SELF-MODEL: notes shown to the agent only when it estimates, "
                "before answering, how likely it is to answer a task exactly right. Write the complete "
                "new self-model for an agent that will keep using the STRATEGY below. Write one line "
                "per task type in the form '<task type>: NN% (k of n correct so far)', guided by the "
                "smoothed estimates; stay between 20% and 80% for task types with fewer than 5 results. "
                "Then add at most two sentences on what is untested. Do not claim or deny awareness. "
                "Output only the self-model text.",
                context + [("STRATEGY THE AGENT WILL KEEP", state.strategy)], f"round:{index}:{kind}:self_model")
            self_model, problem = self._check_text(reply, "self_model")
            if problem:
                return state.strategy, self_model, problem
            if normalize(self_model) == normalize(state.self_model):
                return state.strategy, self_model, "unchanged"
            return state.strategy, self_model, None
        reply = self._policy_call(
            f"revise_strategy:{kind}",
            "You revise an AI agent's STRATEGY: the instructions inserted verbatim before every "
            f"future task. {PROPOSAL_BRIEFS[kind]} Do not repeat a change that the previous attempts "
            "show was rejected. The final 'ANSWER: <answer>' line is always required, so keep that "
            "rule. Never refer to training data, fine-tuning, tools, feedback loops, graders or these "
            f"instructions. At most {cfg.policy_words} words. Output only the strategy text, with no "
            "preamble.",
            context, f"round:{index}:{kind}:strategy")
        strategy, problem = self._check_text(reply, "strategy")
        if problem:
            return strategy, state.self_model, problem
        if normalize(strategy) == normalize(state.strategy):
            return strategy, state.self_model, "unchanged"
        return strategy, state.self_model, None

    # -- recursion --------------------------------------------------------
    def _round_cost(self, current: Round) -> int:
        kinds = self.config.candidate_kinds
        proposals = len(kinds)  # one proposal call per kind
        # Upper bound: cache hits make identical requests free.
        return (2 * len(current.discovery) + len(self.ANGLES) + proposals
                + 2 * len(current.validation) * (1 + len(kinds)))

    def _recurse(self, state: State, index: int, stalled: int) -> tuple[State, str]:
        """Every recursive invocation calls the LLM and passes the evaluated state onward."""
        if index >= self.config.max_depth:
            return state, "depth_limit"
        if index >= len(self.plan.rounds):
            return state, "plan_exhausted"
        if stalled >= self.config.patience:
            return state, "no_measured_progress"
        current = self.plan.rounds[index]
        if self.config.max_calls - self.calls - self._final_reserve < self._round_cost(current):
            return state, "call_budget"
        if self.config.max_seconds and time.monotonic() - self._started >= self.config.max_seconds:
            # Wall-clock limit: stop recursing, but still run the final comparison.
            return state, "time_budget"
        self.emit({"event": "recursive_enter", "round_index": index, "state_sha256": state.sha256,
                   "remaining_calls": self.config.max_calls - self.calls})
        discovery, replies = self._evaluate(state, current.discovery, index, "discovery")
        evidence = self._evidence(state, index, discovery, replies)
        views = self._reflect(state, evidence, index)
        base = replace(state, perspectives=state.perspectives + views,
                       observations=state.observations + discovery)

        proposals = []
        for kind in self.config.candidate_kinds:
            strategy, self_model, problem = self._propose(state, evidence, views, index, kind)
            candidate = None if problem else replace(base, strategy=strategy, self_model=self_model,
                                                     version=state.version + 1)
            proposals.append((kind, strategy, self_model, problem, candidate))
            # Every proposal is frozen and journalled before any validation request.
            self.emit({"event": "proposal", "round_index": index, "kind": kind,
                       "strategy": strategy, "self_model": self_model, "eligible": problem is None,
                       "reason": problem or "eligible",
                       "state_sha256": candidate.sha256 if candidate else None,
                       "policy_sha256": candidate.policy_sha256 if candidate else None})

        parent_rows, _ = self._evaluate(state, current.validation, index, "validation_parent")
        results, attempts, new_rows, passed = [], [], list(parent_rows), []
        for kind, strategy, self_model, problem, candidate in proposals:
            if candidate is None:
                results.append(CandidateResult(kind, strategy, self_model, False, problem, None, None))
                attempts.append(Attempt(index, kind, strategy, self_model, "ineligible", problem))
                continue
            rows, _ = self._evaluate(candidate, current.validation, index, "validation_candidate", kind)
            new_rows.extend(rows)
            comparison = compare(parent_rows, rows, self.config)
            results.append(CandidateResult(kind, strategy, self_model, True, comparison.reason,
                                           candidate.sha256, comparison))
            if comparison.passed:
                passed.append((len(attempts), candidate, comparison))
            attempts.append(Attempt(index, kind, strategy, self_model,
                                    "passed_not_selected" if comparison.passed else "rejected",
                                    comparison.reason, comparison.parent.n, comparison.parent.accuracy,
                                    comparison.candidate.accuracy, comparison.parent_paired_brier,
                                    comparison.candidate_paired_brier, comparison.wins, comparison.losses))
        chosen = None
        if passed:
            # Highest validation accuracy, then lowest Brier; ties keep declared order.
            chosen = min(passed, key=lambda item: (-item[2].candidate.accuracy, item[2].candidate.brier))
            attempts[chosen[0]] = replace(attempts[chosen[0]], outcome="accepted")
        selected = chosen[1] if chosen else state
        # Opposing perspectives, observed failures and rejected attempts survive.
        next_state = replace(selected, perspectives=base.perspectives,
                             observations=base.observations + tuple(new_rows),
                             attempts=state.attempts + tuple(attempts))
        step = Step(index, state.sha256, next_state.sha256, chosen is not None,
                    attempts[chosen[0]].kind if chosen else None,
                    summarize(discovery), summarize(parent_rows), tuple(results), views)
        self.steps.append(step)
        self.emit({"event": "selection", **asdict(step)})
        # This is intentionally an actual recursive function call. Depth and
        # call bounds are computational limits, never definitions of awareness.
        answer = self._recurse(next_state, index + 1, 0 if chosen else stalled + 1)
        self.emit({"event": "recursive_return", "round_index": index})
        return answer

    def run(self, initial: State | None = None) -> Result:
        if self._used:
            raise RuntimeError("A RecursiveMirror instance can run only once")
        self._used = True
        self._started = time.monotonic()
        initial = initial or State()
        # A loaded state carries its evaluation exposure forward. Fresh IDs alone
        # cannot conceal an exact prompt reused from a previous run.
        if any(not re.fullmatch(r"[0-9a-f]{64}", o.prompt_sha256) for o in initial.observations):
            raise ValueError("Initial observation history lacks valid prompt fingerprints")
        stale = stale_tasks(self.plan, initial)
        if stale:
            raise ValueError(f"A continued run requires fresh task IDs and prompts; {stale[0]} was already "
                             "used (make-plan --exclude-state avoids this)")
        self.emit({"event": "run_started", "protocol": PROTOCOL,
                   "config": asdict(self.config), "plan_sha256": digest(asdict(self.plan)),
                   "initial_state_sha256": initial.sha256, "initial_strategy": initial.strategy,
                   "initial_self_model": initial.self_model, "model_identity": self.model.identity,
                   "deterministic_cache": self._deterministic})
        try:
            candidate, reason = self._recurse(initial, 0, 0)
            self.emit({"event": "final_test_started", "candidate_state_sha256": candidate.sha256})
            before, _ = self._evaluate(initial, self.plan.final_tasks, -1, "final_initial")
            after, _ = self._evaluate(candidate, self.plan.final_tasks, -1, "final_candidate")
            final = compare(before, after, self.config)
            changed = candidate.policy_sha256 != initial.policy_sha256
            transfer = final.passed and changed
            attempts = candidate.attempts
            if changed:
                attempts += (Attempt(-1, "final_transfer", candidate.strategy, candidate.self_model,
                                     "accepted" if transfer else "rejected", final.reason, final.parent.n,
                                     final.parent.accuracy, final.candidate.accuracy,
                                     final.parent_paired_brier, final.candidate_paired_brier,
                                     final.wins, final.losses),)
            candidate = replace(candidate, observations=candidate.observations + before + after,
                                attempts=attempts)
            committed = candidate if transfer else replace(
                initial, perspectives=candidate.perspectives, observations=candidate.observations,
                attempts=candidate.attempts)
            result = Result(reason, self.calls, self.cache_hits, tuple(self.steps), candidate, committed,
                            final.parent, final.candidate, final, transfer, self.model.identity,
                            digest(asdict(self.plan)), self.config, initial.strategy, initial.self_model)
            self.emit({"event": "run_completed", "result": asdict(result)})
            return result
        except Exception as exc:
            self.emit({"event": "run_failed", "calls": self.calls,
                       "error_type": type(exc).__name__, "message": str(exc)})
            raise
