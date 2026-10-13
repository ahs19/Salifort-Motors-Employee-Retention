# Digital Mirror: recursive self-improvement core v0.10

The goal of this project is an AI that improves itself. The implementation is
in **`recursive-mirror/`**, and its instructions are in
**`recursive-mirror/README.md`**.

The LLM examines its own measured behaviour from several perspectives and
rewrites its own strategy and self-model. It does this through an actual
recursive function. A change is kept only when it measurably helps on fresh
tasks, and committed only when it also transfers to a held-out task set. The
agent remembers every attempted change and its measured outcome.

**Run the experiment:** open
[`recursive-mirror/release/v0.10.0/Digital_Mirror_Recursive_v0_10.ipynb` in Colab](https://colab.research.google.com/github/ahs19/Salifort-Motors-Employee-Retention/blob/claude/self-improve-ai-revision-3o1opd/recursive-mirror/release/v0.10.0/Digital_Mirror_Recursive_v0_10.ipynb).
Choose a GPU runtime, then *Runtime → Run all*. Upload the downloaded
evidence zip for analysis.

## Version history

- **v0.10 (13-14 October 2026).** Hardened for the first real run:
  - the inference path checked against the real `torch` and `transformers`
    stack;
  - a multi-agent review in which skeptics tried to refute each finding;
  - pure greedy decoding;
  - one lever per proposal kind;
  - batched GPU generation;
  - replicated experiments with a hand-written control arm and pooled
    held-out analysis.

  See `recursive-mirror/VALIDATION.md`.
- **v0.9 (13 October 2026).** Revised v0.8 in response to its failed real
  trials. See `recursive-mirror/docs/v0.8-trial-review.md`.
- **v0.8 (9 October 2026).** The first recursive core. Its first real
  Qwen2.5-3B Colab trials did not improve the agent. That evidence is
  preserved in `recursive-mirror/evidence/v0.8.0-colab-20261009/`, and the
  v0.8 source is the first commit of `recursive-mirror/` in this repository's
  history.

The earlier daily-training design (Awareness Creator revisions 0.2-0.3, in the
`digital-mirror/` folder and its archives) is kept outside this repository as
the earlier experiment. Do not use its commands as the entry point. Like every
later revision, it does not claim to create or detect phenomenal
consciousness.
