# Digital Mirror: current recursive self-improvement core v0.9

The project goal is an AI that improves itself. The current implementation is
in **`recursive-mirror/`**, with instructions in
**`recursive-mirror/README.md`**. The LLM examines its own measured behaviour
from several perspectives, rewrites its own strategy and self-model, and keeps
a change only when it measurably helps on fresh tasks and then on a separate
held-out task set. It does this through an actual recursive function and
remembers every attempted change.

v0.9 (13 October 2026) revises v0.8 using the evidence from its first real
Qwen2.5-3B Colab trials on 9 October 2026, in which v0.8 did not improve
itself. See `recursive-mirror/docs/v0.8-trial-review.md` for the call-by-call
diagnosis and `recursive-mirror/VALIDATION.md` for what has and has not been
verified. The v0.8 trial evidence is preserved in
`recursive-mirror/evidence/v0.8.0-colab-20261009/`.

To run the bounded GPU trial, open
`recursive-mirror/release/v0.9.0/Digital_Mirror_Recursive_v0_9.ipynb` in
Colab with a GPU runtime and run the cells in order.

## Historical record

The v0.8 recursive core (9 October 2026) is the first commit of
`recursive-mirror/` in this repository's history.

The earlier daily-training design (Awareness Creator revisions 0.2-0.3, in the
`digital-mirror/` folder and its archives) is preserved outside this
repository as the earlier experiment. Do not use its commands as the entry
point for v0.9. Revision 0.3 added an optional evidence-grounded functional
self-modelling mode and metacognitive metrics. Like every later revision, it
does not claim to create or detect phenomenal consciousness.
