# Digital Mirror 0.9: recursive self-improvement

The goal of this project is an AI that improves itself. In v0.9 the agent
repeatedly examines its own measured behaviour and rewrites its own working
method. It keeps a change only after the change measurably helps on tasks it
has never seen. The recursion is an actual recursive Python function: each
level calls the LLM, evaluates the result, and passes the improved state into
the next call of itself.

What improves is the agent's **persistent state**:

- **strategy**: the working instructions the agent follows on every task;
- **self-model**: what the agent believes about its own reliability, used
  when it predicts its own success;
- **memory**: every observation, every perspective, and every attempted
  change with its measured result.

Neural weights stay fixed. The LLM cannot rewrite the evaluator, scores,
budgets, tasks or answer keys. Subjective awareness remains a research
hypothesis: a depth limit, eloquent reflection, agreement or a stable state
does not certify it.

## Why v0.9

v0.8 ran on a real model (Qwen2.5-3B on a Colab T4) on 9 October 2026 and did
not improve itself. The first trial aborted when one reflection hit the token
limit. In the second trial the model proposed "expand the training dataset"
as its strategy, which it could not carry out and which made its answers worse
(38 + 47 → 125). A proposal was also "accepted" only because the parent's
forecast was malformed. [`docs/v0.8-trial-review.md`](docs/v0.8-trial-review.md)
traces each failure to its cause, call by call, and maps it to the v0.9 change.
In short, v0.9:

1. tells the model exactly which levers exist and how its text will be used;
2. lets strategies change how the agent works (reasoning is allowed; the
   harness extracts a final `ANSWER:` line);
3. tests the strategy where it acts (answers) and the self-model where it acts
   (forecasts);
4. remembers every attempted change and its measured result, so later levels
   can learn from earlier ones;
5. proposes several kinds of candidate per level and selects among them;
6. compares calibration only on paired valid forecasts, so a formatting
   failure cannot pose as an improvement;
7. survives truncated or empty outputs without aborting the experiment;
8. ships task plans with real headroom, because the v0.8 canary was already
   solved at 100%.

## The recursive self-improvement loop

`recursive_mirror/core.py:RecursiveMirror._recurse` runs one level:

1. **Act.** On fresh discovery tasks, the agent first forecasts its
   probability of being exactly right (one integer, 0-100), then answers. It
   may reason; it finishes with `ANSWER: <answer>`. The harness scores the
   extracted answer by deterministic exact match.
2. **Reflect from three angles.** The *actor* explains each miss, the
   *reviewer* checks calibration and earlier attempts, and the *whole system*
   view integrates both and names the most valuable change. Each sees the
   harness evidence: every discovery task, the agent's answer, the
   reference answer and a reasoning excerpt for each miss, per-category
   calibration of the current policy, the recent attempt ledger and earlier
   views.
3. **Propose.** Each configured kind produces a complete candidate policy:
   - `repair`: a new strategy that fixes the observed errors, plus a
     matching self-model;
   - `rewrite`: a substantially different strategy (optional);
   - `recalibrate`: the same strategy with a corrected self-model.

   A candidate that is truncated, empty, over-long or unchanged is recorded
   as ineligible and costs no validation calls.
4. **Freeze, then validate.** Every proposal is journalled before any
   validation request. The parent and each candidate run on the same fresh
   validation tasks with paired seeds. Validation answer keys never reach
   the model.
5. **Select.** A candidate passes the gate (`core.compare`) when:
   - accuracy does not fall, and overall Brier loss does not rise. An invalid
     forecast counts as the worst possible loss, so malformed forecasts can
     never hide miscalibration;
   - it shows at least one **consistent** gain. Either accuracy rises by
     `min_accuracy_gain`, or Brier loss falls by `min_brier_gain` on tasks
     where *both* arms forecast validly (with no loss of forecast coverage).
     In both cases the paired one-sided p-value across tasks must be at most
     `max_p_value` (default 0.25).

   The best passing candidate becomes the next state. Every candidate,
   accepted or not, enters the attempt ledger.

   The p-value bound matters. Without it, an irrelevant strategy that merely
   reshuffles answers gains one net task, and passes, about 40% of the time
   on 8-12 tasks. With it, about 13-15% of such candidates pass, while about
   70-85% of genuine +40-point improvements still do (simulated with
   independent arms). The final held-out test applies the same gate again.
6. **Recurse.** `_recurse(next_state, level + 1, …)` is called directly. It
   stops at the depth limit, the call budget (the final test is always
   reserved), the end of the plan, or after `patience` levels without an
   accepted change.

After the recursion returns, the selected policy and the initial policy run on
a separate final task set. The new policy is committed only if it passes the
same gate there. Otherwise the strategy and self-model roll back, while all
memory (observations, perspectives, attempts, including the failed transfer)
is kept for the next run.

### Deterministic replay

The Hugging Face backend uses greedy decoding and declares itself
deterministic. Within one run, a byte-identical request then reuses its first
reply instead of calling the model again. For example, a `recalibrate`
candidate keeps the parent's strategy, so its answer requests are identical to
the parent's. Paired comparisons stay exact, and calls are saved (reported as
`cache_hits`). Backends that sample are never cached.

## Paradox and perspective

Here "paradox" keeps the project's meaning: different angles on the same
system can yield different descriptions. The harness preserves each
description with its angle, level and state identity, flags incomplete ones,
and never demands consensus. All three views come from the same LLM: they are
perspectives, not independent witnesses. Only the accepted strategy and
self-model enter task-solving prompts, so keeping a rejected view for
examination never deploys it.

## Running it

The core and its tests need Python 3.11+ and the standard library only. From
this directory:

```powershell
python -m unittest discover -s tests -v
python -m recursive_mirror demo --output runs/demo-001
```

The demo is a scripted test double programmed to improve, so that every
selection path is exercised. It is **not** LLM evidence.

Generate a task plan with headroom, or use the shipped
`examples/headroom.json` (seed 2026: 2 levels × (6 discovery + 10 validation)
plus 16 final tasks):

```powershell
python -m recursive_mirror make-plan --output plans/seed7.json --seed 7 --rounds 3 --discovery 6 --validation 10 --final 20
```

Run a real model (install `torch`, `transformers`, `accelerate` and, for
`--four-bit`, `bitsandbytes` first):

```powershell
python -m recursive_mirror run --plan examples/headroom.json --output runs/real-001 --model C:\models\Qwen2.5-3B-Instruct --device cuda --four-bit --max-depth 2 --max-calls 240
```

Useful options:

| option | default | meaning |
|---|---|---|
| `--candidates` | `repair,recalibrate` | proposal kinds per level (`repair`, `rewrite`, `recalibrate`) |
| `--max-depth`, `--patience` | 4, 2 | recursion depth, levels without progress before stopping |
| `--max-calls` | 400 | hard LLM call budget (final test reserved) |
| `--answer-tokens` | 512 | output budget for answers (reasoning included) |
| `--max-p-value` | 0.25 | evidence bound on every accepted gain (1 disables it) |
| `--max-input-tokens` | 8192 | longer prompts are refused, never truncated (v0.8 used 4096) |
| `--hide-discovery-answers` | off | do not show reference answers for missed discovery tasks |
| `--initial-strategy`, `--initial-self-model` | defaults | start from a hand-written policy (a control run) |
| `--state` | none | continue from a checkpoint (same model identity, fresh tasks) |

The call budget for one level is at most `2·discovery + 3 + proposal calls +
2·validation·(1 + candidates)`, plus `4·final` once. The shipped plan needs at
most 220 calls for two levels.

Remote downloads require `--allow-download` and an immutable 40-character
`--revision`. A CUDA run must actually have CUDA. Each real call records its
input/output token counts, revision, device and dtype. These receipts do not
independently verify model-weight bytes.

**Continuing.** `--state runs/real-001/checkpoint.json` carries the committed
policy and all memory forward. Supply a fresh plan, for example from
`make-plan` with a new seed: a reused task ID or normalized prompt is
rejected. The checkpoint hash detects accidental modification; it is not a
signature. v0.8 checkpoints (format v1) still load. Their backend identity
must match, which for v0.8 runs means `--max-input-tokens 4096`.

**Colab.** `python build_release.py` writes a verified source archive and
`Digital_Mirror_Recursive_v0_9.ipynb`. The notebook checks the archive, runs
the tests and the demo, then runs the pinned Qwen2.5-3B self-improvement
trial on the headroom plan. An optional control run starts from a hand-written
step-by-step strategy. A final cell exports all evidence, including from
failed runs.

## Outputs

Every run uses a new directory:

- `journal.jsonl`: every request, raw response, truncation, cache hit,
  observation, proposal, selection and failure, flushed as it happens;
- `plan.json`: the exact plan used;
- `result.json`: per-level comparisons, the final comparison and both states;
- `checkpoint.json`: the committed state, hash-checked (format v2);
- `report.md`: a readable summary covering what was tried at each level, what
  passed, the final held-out comparison and the committed policy.

A failed run leaves its journal and publishes no result or checkpoint.

## Reading a result honestly

- `transfer_improved: true` means the self-revised policy beat the initial
  policy on held-out tasks under the declared gate, for this model and plan.
  It does not mean general self-improvement.
- The gates are exploratory. The p-value bound is not corrected for the
  several candidates and levels tried, and 0.25 is lenient by design so that
  real gains on small task sets can pass. With 16 final tasks, even a clear
  gain is uncertain. Repeat with new plans (new seeds) before drawing
  conclusions.
- Improvement is measured relative to the starting policy. The default start
  ("return only the requested answer") is deliberately plain. Use
  `--initial-strategy` with a good hand-written prompt to test whether the
  loop adds anything beyond "think step by step".
- Selecting the best of several candidates on validation inflates validation
  scores. Only the final comparison should be read as evidence.
- Revealing reference answers for missed discovery tasks is a deliberate
  learning signal. Discovery tasks are never used for selection; validation
  and final keys are never shown.
- Exact-prompt disjointness cannot detect paraphrases or leakage from the
  model's pre-training data.

## Next measurements

1. Run the shipped notebook trial and its control on the T4, then repeat with
   fresh `make-plan` seeds.
2. Continue a committed checkpoint on a new plan to test whether gains
   accumulate across runs.
3. Use the verified-correct reasoning traces in the journal as candidate
   training data for a separate, weight-level self-training stage. That stage
   would need its own held-out evaluation and is not implemented here.
