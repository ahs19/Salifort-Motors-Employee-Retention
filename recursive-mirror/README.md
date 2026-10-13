# Digital Mirror 0.8 — recursive self-modelling

This is the new project core requested on 9 October 2026. The harness calls an
LLM from an actual recursive Python function. Each level examines the system's
behaviour, considers several perspectives, proposes a new strategy and self-model,
tests them, and passes the resulting state into the next recursive call.

The working question is whether recursive self-examination can produce persistent
improvements in an agent's strategy and predictions about its own performance.
Subjective awareness remains a research hypothesis. A depth limit, eloquent
reflection, agreement or a stable state does not certify it.

The earlier daily fine-tuning experiment remains in the sibling directories and
archives. Its failed format tests are historical evidence, not results for v0.8.

## The recursive core

`recursive_mirror/core.py:RecursiveMirror._recurse` implements this sequence:

1. Run fresh discovery tasks using the current strategy and self-model. Before
   each answer, ask the LLM to forecast its probability of getting it right.
2. Call the LLM from three angles: the acting model, its reviewer and the whole
   system. The harness attaches the state identity and context to every view.
3. Ask for a revised strategy and a revised self-model in ordinary text.
4. Freeze that proposal, then compare it with the parent on the same fresh
   validation tasks. The LLM never receives the answer keys. These tasks become
   selection evidence once used; the next level receives different tasks.
5. Accept the proposal only if accuracy and forecast calibration do not worsen
   and at least one improves by the declared minimum. Otherwise retain the
   previous strategy and self-model. Retain all perspectives and observations
   in either case.
6. Call `_recurse(next_state, next_depth, ...)` directly. Stop at the declared
   depth, call budget, end of the task plan or repeated absence of progress.

After recursion returns, compare the selected state with the original state on
a separate final task set. A successful final comparison is required before the
new strategy and self-model become the committed checkpoint. Otherwise those
fields roll back to the original; the complete evidence and perspectives survive.

This is improvement of the persistent **agent state**. Neural weights stay fixed
in this version. There is no nightly training stage or autonomous code execution.
The LLM cannot rewrite the evaluator, scores, budgets or answer keys.

## Paradox and perspective

Here, “paradox” follows the project's chosen meaning: different angles on the same
system can yield different descriptions. The harness preserves those descriptions
with their observer angle, recursive level and state identity. It neither demands
consensus nor deletes disagreement when a candidate is rejected.

The next reflection sees the latest three prior views; the full ledger remains
in the checkpoint and journal. Only the accepted strategy and self-model enter
task-solving prompts. Thus keeping a rejected view for examination cannot silently
deploy that view as the agent's new policy.

The same LLM supplies all three views by default. These are perspectives, not
independent scientific witnesses. Statements about evidence remain hypotheses
until measured. Preserving a contradiction does not make both claims established
facts under the same conditions.

## Removing the old format barrier

The LLM does not fill the old nested awareness JSON form, copy hashes or invent
issue identifiers. Its strategy, self-model, perspectives and task answers are
plain text. The harness generates all structured records.

The forecast is deliberately small: one integer from 0 to 100. An invalid forecast
is retained as unscored text, receives the worst calibration loss, and prevents
scientific acceptance of that candidate batch. It is never repaired into a valid
forecast. Empty responses, failed transport, input overflow and truncated
generations fail with an audit record.

Calibration uses mean Brier loss: the squared difference between predicted
probability and observed correctness. Smaller is better. Tasks use deterministic
exact-answer scoring after case/whitespace normalization. This evaluator supports
short objective tasks; open-ended answers need a separately defined evaluator.

## Local checks and use

The core and its unit tests use Python 3.11+ and the standard library. From this
directory:

```powershell
python -m unittest discover -s tests -v
python -m recursive_mirror demo --output runs/demo-001
```

The demo is a scripted test double. Its apparent improvement is programmed to
exercise selection and persistence. It is not LLM evidence.

For a real model, install the optional inference dependencies in the chosen
runtime and provide a model path already present there:

```powershell
python -m recursive_mirror run --plan examples/canary.json --output runs/real-001 --model C:\models\Qwen2.5-3B-Instruct --device cuda --four-bit --max-depth 2 --max-calls 40
```

Remote downloads require explicit `--allow-download` and `--revision` with an
immutable 40-character commit hash, shared by tokenizer and model. A CUDA run must actually
have CUDA available. The backend uses greedy generation, a declared per-call
output bound, a 4096-token input limit and native FP16 on T4-class hardware. Long
input is refused rather than silently truncated. Model loading occurs only in
the real `run` command, not import, unit tests or the demo. Each real call records
input/output token counts, its requested/resolved revision and actual device/dtype.
These receipts do not independently verify model-weight bytes.

`--state path/to/checkpoint.json` continues from a saved checkpoint. Supply a fresh
task plan: reuse of a previous task ID or normalized prompt is rejected. The
checkpoint's hash detects accidental modification; it is not an authentication
signature. CLI continuation also requires the same backend identity, including
model revision, device choice and quantization. Keep its journal and original
run metadata alongside it. Checkpoint save/load share a 16 MiB document limit and
10,000-item limits for each history collection. Exceeding a limit stops publication
with the journal retained; it never silently trims perspectives or evidence.

Every run uses a new output directory. Raw requests, responses, model errors,
observations, proposals, selections and completion are flushed to a journal.
Results and a checkpoint are written at completion. A failed model call leaves
its evidence available without publishing an improved checkpoint.

## Colab trial

`build_release.py` builds a source archive and a separate Colab notebook from
explicit package files. The notebook verifies the archive before import, runs
the unit checks and scripted demo, then offers a bounded real-model GPU trial.
It uses a new directory so older experiment evidence stays separate.

The supplied arithmetic canary is a plumbing check with two recursive levels,
one discovery and one validation task per level, plus two final tasks. At most
30 LLM calls are needed: `2*discovery + 5 + 4*validation` per level and
`4*final_tasks` for the final comparison. It is too small and too simple to
establish general self-improvement or awareness. A perfect initial score may
leave no room for task-accuracy improvement.

## Scientific limits and next measurements

The first useful result is whether the complete recursive process works with a
real LLM and whether any selected change transfers to the final tasks. A stronger
study needs varied held-out tasks, repeated seeds, uncertainty estimates and
matched non-recursive/self-review controls. Those controls are not implemented
or claimed by this minimal new core.

Acceptance thresholds are engineering gates for this pilot, not significance
tests. Greedy decoding and matched seeds do not guarantee identical results on
all hardware. Exact prompt disjointness cannot detect paraphrases or leakage
from a model's original training data. Repeatedly tuning against final results
would turn that final set into development data; use a new plan for the next run.
