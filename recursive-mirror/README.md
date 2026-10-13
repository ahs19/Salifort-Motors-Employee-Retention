# Digital Mirror 0.10: recursive self-improvement

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

1. Open [`release/v0.10.0/Digital_Mirror_Recursive_v0_10.ipynb` in Colab](https://colab.research.google.com/github/ahs19/Salifort-Motors-Employee-Retention/blob/claude/self-improve-ai-revision-3o1opd/recursive-mirror/release/v0.10.0/Digital_Mirror_Recursive_v0_10.ipynb).
   For a private repository, Colab first asks you to authorise GitHub.
2. *Runtime → Change runtime type →* a GPU (T4 works; L4 or A100 is faster).
3. *Runtime → Run all*, and keep the tab open.

The last cell downloads `…_evidence.zip`. If any cell stops with an error, run
that last cell by hand: it exports everything produced so far. The default
settings take about 2 hours on a T4.

The notebook works through these steps in order:

1. verifies the embedded, hash-checked source archive;
2. runs the unit tests;
3. checks for a GPU and installs the pinned `transformers==4.57.6`;
4. tests the real inference stack on the GPU with a tiny locally built model;
5. runs a short preflight with the real model (Qwen2.5-3B-Instruct, pinned
   revision) and prints protocol diagnostics;
6. runs the experiment (three task plans × self arm and control arm);
7. writes a pooled `summary.md`.

## Why v0.10

**v0.8 → v0.9.** v0.8 ran on the real model on 9 October 2026 and did not
improve itself. [`docs/v0.8-trial-review.md`](docs/v0.8-trial-review.md) traces
each failure call by call:

- one truncated reflection aborted the first trial;
- the model proposed "expand the training dataset" as its strategy, which
  made its answers worse;
- one proposal was "accepted" only because the parent's forecast was
  malformed.

**v0.9 → v0.10.** Before any GPU time was spent, the whole inference path was
run against the real `torch` and `transformers` libraries. A multi-agent
review with adversarial checks also went through every protocol detail. They
found:

- **Decoding was not truly greedy.** Qwen2.5 ships `repetition_penalty: 1.05`,
  and transformers applies it even with `do_sample=False`. Every digit or
  letter already in the prompt was penalised. Decoding is now plain argmax,
  and the receipts record it.
- **One lever per proposal.** A long self-model reply (v0.8's were 1541
  characters) could discard a good strategy. A strategy was also forecast
  with a self-model that described the *old* strategy, so calibration could
  veto a real accuracy gain. Now `repair` and `rewrite` change only the
  strategy and are judged on answers. `recalibrate` changes only the
  self-model and is judged on forecasts.
- **Forecasts that are task answers.** A forecast equal to the task's answer
  (v0.8 answered "216" when asked for a probability) is now invalid.
  Forecasts are requested as percentages.
- **Answer extraction** now handles Qwen's habits (`<…>`, `$…$`, `\boxed{}`, a
  trailing "answer:"). The same format-only rules apply to every arm.
- **Prompt hygiene.** Hashes, backend identities and budgets are no longer
  shown to the model, which had copied them into its reflections. Reflections
  now get up to 1500 characters of the agent's reasoning for missed tasks and
  smoothed per-type calibration, and the repair brief is evidence-led. Advice
  about training or fine-tuning is screened out before any validation calls.
- **Batched generation.** Independent evaluation requests run in GPU batches.
  This is several times faster, which pays for larger task sets and stronger
  statistics. On the real library, batched rows match single requests
  exactly on CPU float32.
- **Replicated experiments** with a hand-written control arm and pooled
  held-out analysis (`experiment`, `summarize`), plus `diagnose` for
  protocol health. A wall-clock budget lets a slow GPU still finish with a
  report, and a preflight runs before the long run.

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
   - `repair` writes a new strategy that fixes the observed errors.
   - `rewrite` writes a substantially different strategy (optional).
   - `recalibrate` writes a new self-model for the unchanged strategy.

   Unchanged, empty, truncated, over-long or non-executable proposals are
   recorded as ineligible and cost no validation calls.
4. **Freeze, then validate.** Every proposal is journalled before any
   validation request. The parent and each candidate run on the same fresh
   validation tasks with paired seeds. Validation answer keys never reach
   the model.
5. **Select (`core.compare`).** Accuracy is the primary objective.
   - Accuracy may never fall.
   - An accuracy gain passes when it is at least `min_accuracy_gain` and its
     one-sided paired p-value across tasks is at most `max_p_value` (default
     0.25).
   - Otherwise a calibration gain passes when all of these hold: the Brier
     loss falls on tasks where *both* arms forecast validly, with the same
     p-value bound; forecast coverage does not drop; and overall Brier does
     not rise (an invalid forecast counts as the worst loss).

   The best passing candidate becomes the next state. Every candidate enters
   the attempt ledger.
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
accuracy per task type. By default each seed's plan has 3 levels of 12
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
python -m recursive_mirror make-plan --output plans/seed7.json --seed 7 --rounds 3 --discovery 12 --validation 24 --final 48
python -m recursive_mirror make-plan --output plans/next.json --seed 8 --exclude-state runs/real-001/checkpoint.json
python -m recursive_mirror run --plan examples/headroom.json --output runs/real-001 --model Qwen/Qwen2.5-3B-Instruct --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 --allow-download --device cuda --max-depth 2 --max-calls 240
python -m recursive_mirror experiment --output runs/exp-001 --model Qwen/Qwen2.5-3B-Instruct --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 --allow-download --device cuda --seeds 2026,2027,2028 --max-depth 3 --max-calls 720
python -m recursive_mirror summarize runs/exp-001
python -m recursive_mirror diagnose runs/exp-001/seed-2026/self
```

| option | default | meaning |
|---|---|---|
| `--candidates` | `repair,recalibrate` | proposal kinds per level (`repair`, `rewrite`, `recalibrate`) |
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

1. Run the notebook and analyse the evidence zip.
2. If the loop transfers, continue committed checkpoints on fresh plans to
   test whether gains accumulate.
3. Use the verified-correct reasoning traces in the journals as candidate
   training data for a separate, weight-level self-training stage. That stage
   would need its own held-out evaluation and is not implemented here.
