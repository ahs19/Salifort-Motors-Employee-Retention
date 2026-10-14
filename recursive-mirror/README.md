# Digital Mirror 0.11: recursive self-improvement

The goal of this project is an AI that improves itself. The agent repeatedly
examines its own measured behaviour from several perspectives and rewrites its
own working method. It keeps a change only when the change measurably helps on
fresh tasks, and commits it only when the gain also transfers to held-out tasks
that played no part in any decision. The recursion is an actual recursive
Python function: each level calls the LLM, evaluates the result, and passes the
improved state into the next call of itself.

What improves is the agent's **persistent state**:

- **strategy**: the working instructions the agent follows on every task;
- **self-model**: what the agent believes about its own reliability, used
  when it predicts its own success before answering;
- **memory**: every observation, every perspective, and every attempted
  change with its measured result.

Neural weights stay fixed. The LLM cannot rewrite the evaluator, scores,
budgets, tasks or answer keys. Subjective awareness remains a research
hypothesis: a depth limit, eloquent reflection, agreement or a stable state
does not certify it.

## Run the experiment on Colab

1. Open [`release/v0.11.0/Digital_Mirror_Recursive_v0_11.ipynb` in Colab](https://colab.research.google.com/github/ahs19/Salifort-Motors-Employee-Retention/blob/claude/self-improve-ai-revision-3o1opd/recursive-mirror/release/v0.11.0/Digital_Mirror_Recursive_v0_11.ipynb).
   For a private repository, Colab first asks you to authorise GitHub.
2. *Runtime → Change runtime type →* a GPU (T4 works; L4 or A100 is faster).
3. *Runtime → Run all*, and keep the tab open.

The last cell downloads `…_evidence.zip`. If any cell stops with an error, run
that last cell by hand: it exports everything produced so far. The default
settings take about 40 minutes on an A100 and several hours on a T4 (use fewer
`SEEDS` or `ROUNDS` there).

The notebook works through these steps in order:

1. verifies the embedded, hash-checked source archive;
2. runs the unit tests;
3. checks for a GPU and installs the pinned `transformers==4.57.6`;
4. tests the real inference stack on the GPU with a tiny locally built model;
5. runs a short preflight with the real model (Qwen2.5-3B-Instruct, pinned
   revision) and prints protocol diagnostics;
6. runs the experiment (three task plans × self arm and control arm);
7. writes a pooled `summary.md`.

## Results so far

The first real-model run (v0.10, Qwen2.5-3B-Instruct on a Colab A100,
14 October 2026, three task plans × two arms, 144 pooled held-out tasks) is
analysed in [`docs/v0.10-a100-results.md`](docs/v0.10-a100-results.md), with
the full evidence in `evidence/v0.10.0-colab-a100-20261014/`.

- **The self-model improved itself and the gain transferred.** In all six
  runs the agent's own rewritten self-model forecast its held-out success
  better than its starting one: pooled Brier loss 0.328 → 0.198 (p < 0.001).
- **The strategy improved itself once.** In one of three plans the agent
  wrote itself a verification rule for multiplication. It passed validation,
  and held-out accuracy rose from 60% to 77% (9 wins, 1 loss, p ≈ 0.011).
  Held-out multiplication went from 2 of 8 to 8 of 8.
- **It did not beat a hand-written step-by-step prompt** (70% vs 73%).
- **Whole-strategy rewrites by the 3B model mostly made things worse.** 11 of
  18 `repair` proposals lowered accuracy, often by dropping instructions that
  already worked or by copying the evidence they were shown.

## Why v0.11

v0.11 builds on what worked in that run, a single added rule:

- **`add_rule` and `add_rule_2` replace whole-strategy rewrites by default.**
  The agent writes *one* new rule for its weakest (or second-weakest) task
  type, and the harness appends it to the current strategy. Rules that
  already work are never lost, accepted rules accumulate over levels, and
  each change is small enough for a paired test to judge. The proposer sees a
  short, focused context (about 1.3k characters instead of about 12k): the
  target task type, up to three missed discovery examples with the correct
  answer and its own working, the rules already tried with their outcomes,
  and its own view of the misses.
- **Screens before any validation calls.** A rule that talks about forecasts
  or confidence, copies harness sections or refers to "the reference answer"
  is ineligible (`rule_off_lever`). So is a whole-strategy rewrite that copies
  harness sections (`strategy_echoes_context`).
- **Structured self-model.** Only the `<task type>: …` lines of a
  recalibration reply are kept. Copied evidence or truncation after those
  lines no longer discards a good self-model (4 of 18 were lost in the A100
  run).
- **Four levels per run** (1100 calls per run in the notebook).

### Earlier revisions

- **v0.8 → v0.9.** v0.8's first real trials (9 October 2026) did not improve
  the agent. [`docs/v0.8-trial-review.md`](docs/v0.8-trial-review.md) traces
  each failure call by call: a truncated reflection aborted a run, the model
  proposed "expand the training dataset" as its strategy, and one proposal
  was accepted only because the parent's forecast was malformed.
- **v0.9 → v0.10.** The inference path was run against the real `torch` and
  `transformers` libraries, and a multi-agent review with adversarial checks
  went through every protocol detail. That found and fixed:
  - non-greedy decoding (Qwen2.5's shipped repetition penalty);
  - proposals that moved two levers at once;
  - forecasts that were really task answers;
  - answer-extraction misses;
  - harness bookkeeping leaking into prompts.

  It also added batched generation, replicated experiments with a control
  arm, `diagnose`, a preflight and a wall-clock budget. See
  [`VALIDATION.md`](VALIDATION.md).

## The recursive self-improvement loop

`recursive_mirror/core.py:RecursiveMirror._recurse` runs one level:

1. **Act.** On fresh discovery tasks, the agent first forecasts its
   probability of being exactly right, then answers. It may reason, and it
   finishes with `ANSWER: <answer>`. The harness scores the extracted answer
   by deterministic exact match.
2. **Reflect from three angles.** The *actor* explains each miss. The
   *reviewer* checks calibration and earlier attempts. The *whole-system*
   view integrates both, says where they conflict and in which context each
   holds, and names the most valuable change. All three see the evidence:
   - each discovery task, the agent's answer and the reference answer;
   - a reasoning excerpt for each miss;
   - smoothed calibration of the current policy per task type;
   - recent attempts with their measured outcomes;
   - earlier views.
3. **Propose, one lever per kind.**
   - `add_rule` writes one rule for the task type the current policy handles
     worst, ranked by smoothed accuracy, and appends it to the strategy.
   - `add_rule_2` does the same for the second-worst task type.
   - `recalibrate` writes a new self-model for the unchanged strategy.
   - `repair` and `rewrite` replace the whole strategy (optional; v0.10's
     defaults).

   Unchanged, empty, truncated, over-long, off-lever or non-executable
   proposals are recorded as ineligible and cost no validation calls.
4. **Freeze, then validate.** Every proposal is journalled before any
   validation request. The parent and each candidate run on the same fresh
   validation tasks with paired seeds. Validation answer keys never reach
   the model.
5. **Select (`core.compare`).** Each changed text is judged where it acts,
   and accuracy may never fall.
   - **Accuracy route** (when the strategy changed): the gain must be at
     least `min_accuracy_gain`, with a one-sided paired p-value across tasks
     of at most `max_p_value` (default 0.25).
   - **Calibration route** (when the self-model changed): Brier loss must fall
     on tasks where *both* arms forecast validly, with the same p-value bound,
     and overall Brier must not rise (an invalid forecast counts as the worst
     loss). A drop in forecast coverage is allowed only if that overall Brier
     itself clearly falls.
   - A strategy-only proposal is therefore never accepted for nicer-looking
     forecasts.
   - A composite change, as in the final test, may pass on either route. A
     non-significant accuracy change never blocks a real calibration gain.

   The best passing candidate becomes the next state. Every candidate enters
   the attempt ledger. Untested proposals are shown to later levels as
   "NOT TESTED", not as rejected.
6. **Recurse.** `_recurse(next_state, level + 1, …)` is called directly. It
   stops at the depth limit, the call or time budget (the final test is
   always reserved), the end of the plan, or after `patience` levels without
   progress.

After the recursion returns, the selected policy and the initial policy run on
a separate final task set. The new policy is committed only if it passes the
same gate there. Otherwise the strategy and self-model roll back, while all
memory (observations, perspectives, attempts, including the failed transfer)
is kept for the next run.

The p-value bound matters. Without it, an irrelevant strategy that merely
reshuffles answers gains one net task, and passes, about 40% of the time on
8-12 tasks. With it, about 13-15% of such candidates pass, while about 70-85%
of genuine +40-point improvements still do (simulated with independent arms).
The held-out final test applies the same gate again.

**Deterministic replay.** Greedy decoding is declared deterministic, so within
a run a byte-identical request reuses its first reply. A `recalibrate`
candidate therefore gets exactly the parent's answers, and an unchanged policy
costs nothing in the final test. Batched generation can round slightly
differently from single requests on a GPU, but it stays deterministic for a
given batch.

## Experiment design

`python -m recursive_mirror experiment` runs every seed's independently
generated plan twice. The *self* arm starts from the plain default policy. The
*control* arm starts from a hand-written step-by-step prompt. `summary.md`
then pools the held-out final tasks across seeds to answer three questions:

1. **Self vs initial:** did the policy the loop selected beat the policy it
   started from?
2. **Self vs control:** did it beat a good hand-written prompt on the same
   tasks? This separates "the loop discovered something" from "any
   step-by-step instruction helps".
3. **Control self-improvement:** can the loop improve even that hand-written
   prompt?

Each run also has its own `report.md`, and `summary.md` includes held-out
accuracy per task type. By default each seed's plan has 4 levels of 12
discovery and 24 validation tasks, plus 48 final tasks, so three seeds pool
144 held-out tasks per comparison.

These sizes come from a power analysis by one of the reviewers. For a
genuine +15-point improvement, the chance of passing the gate at least once
in a run is about 0.79, and the final test passes it with power about 0.88.
At 12 validation and 24 final tasks, the chance of committing it was only
about 0.49. Task plans come from `make-plan`: exact-answer tasks
with headroom for small models (multi-digit multiplication, letter counting,
word reversal, decimal comparison, multi-step arithmetic, weekday offsets).
Python computes every answer, and every plan is disjoint across its splits.

## Running locally

The core and its tests need Python 3.11+ and the standard library only:

```powershell
python -m unittest discover -s tests -v
python -m recursive_mirror demo --output runs/demo-001
```

The demo is a scripted test double programmed to improve. It is not LLM
evidence. The real-stack integration tests are opt-in: with `torch`,
`transformers` and `tokenizers` installed, set
`RECURSIVE_MIRROR_TEST_DEVICE=cpu` (or `cuda`).

```powershell
python -m recursive_mirror make-plan --output plans/seed7.json --seed 7 --rounds 4 --discovery 12 --validation 24 --final 48
python -m recursive_mirror make-plan --output plans/next.json --seed 8 --exclude-state runs/real-001/checkpoint.json
python -m recursive_mirror run --plan examples/headroom.json --output runs/real-001 --model Qwen/Qwen2.5-3B-Instruct --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 --allow-download --device cuda --max-depth 2 --max-calls 280
python -m recursive_mirror experiment --output runs/exp-001 --model Qwen/Qwen2.5-3B-Instruct --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 --allow-download --device cuda --seeds 2026,2027,2028 --max-depth 4 --max-calls 1100 --dtype bfloat16
python -m recursive_mirror summarize runs/exp-001
python -m recursive_mirror diagnose runs/exp-001/seed-2026/self
```

| option | default | meaning |
|---|---|---|
| `--candidates` | `add_rule,add_rule_2,recalibrate` | proposal kinds per level (`add_rule`, `add_rule_2`, `recalibrate`, `repair`, `rewrite`) |
| `--max-depth`, `--patience` | 4, 2 | recursion depth; levels without progress before stopping |
| `--max-calls`, `--max-seconds` | 400, 0 | per-run call and wall-clock budgets (the final test is always reserved) |
| `--batch-size` | 8 | independent evaluation requests generated together |
| `--answer-tokens` | 512 | output budget for answers (reasoning included) |
| `--max-p-value` | 0.25 | evidence bound on every accepted gain (1 disables it) |
| `--dtype` | auto | auto = float16 on CUDA, float32 on CPU; bfloat16 suits L4/A100 |
| `--max-input-tokens` | 8192 | longer prompts are refused, never truncated |
| `--hide-discovery-answers` | off | do not show reference answers for missed discovery tasks |
| `--initial-strategy`, `--initial-self-model` | defaults | `run` only: start from a hand-written policy |
| `--state` | none | `run` only: continue from a checkpoint (same model identity, fresh tasks) |

Remote downloads require `--allow-download` and an immutable 40-character
`--revision`. Each real call records its input and output tokens, revision,
device, dtype, decoding and generation time. These receipts do not
independently verify model-weight bytes. Checkpoints are hash-checked (format
v2, which adds the attempt ledger); v1 checkpoints from v0.8 still load. To
continue a committed checkpoint, generate the next plan with `make-plan
--exclude-state <checkpoint>` so that no prompt repeats. A stale plan is
refused before any run directory is created. Their
backend identity differs, because v0.8 decoding was not purely greedy, so they
continue through the Python API rather than the CLI.

`python build_release.py --output release/vX.Y.Z` rebuilds the source archive
and notebook. A test fails if the committed release does not embed exactly
the current sources.

## Outputs

Every run uses a new directory:

- `journal.jsonl`: every request, raw response, truncation, cache hit,
  observation, proposal, selection and failure, flushed as it happens with
  elapsed time;
- `plan.json`, `result.json` and `checkpoint.json`;
- `report.md`: what was tried at each level, what passed, the final held-out
  comparison and the committed policy.

An experiment directory adds `experiment.json`, `summary.md` and
`summary.json`, and a `failure.txt` for any run that crashed. The other runs
still complete. A failed run publishes no result or checkpoint.

## Reading a result honestly

- A positive pooled *self vs initial* result means the self-revised policy
  answered held-out tasks better than its starting policy, for this model and
  these task types. It does not mean general self-improvement.
- The *self vs control* row decides whether the loop added anything beyond a
  generic step-by-step prompt. Without that, a gain may simply be
  "reasoning helps".
- The gates are exploratory. The p-value bound is not corrected for the
  several candidates, levels and comparisons, and 0.25 is lenient by design
  so that real gains on small task sets can pass. The pooled p-values over
  all seeds are the better guide.
- Selecting the best of several candidates on validation inflates validation
  scores. Read the held-out final comparisons.
- Revealing reference answers for missed discovery tasks is a deliberate
  learning signal. Discovery tasks are never used for selection, and
  validation and final keys are never shown.
- Exact-prompt disjointness cannot detect paraphrases or leakage from the
  model's pre-training data.

## Next steps

1. Run the v0.11 notebook and compare it with the v0.10 A100 run. The test
   is whether added rules raise held-out accuracy in more than one of three
   plans, and whether the self arm closes the gap to the hand-written prompt.
2. If the loop transfers, continue committed checkpoints on fresh plans to
   test whether gains accumulate.
3. Use the verified-correct reasoning traces in the journals as candidate
   training data for a separate, weight-level self-training stage. That stage
   would need its own held-out evaluation and is not implemented here.
