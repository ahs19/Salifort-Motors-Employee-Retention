# v0.10 validation (13-14 October 2026)

**Status:** ready for its first real-model run, which has not happened yet.
Every part of the inference path was exercised against the real
`torch 2.14.1` and `transformers 4.57.6` libraries, with a tiny locally built
Qwen2-architecture model on CPU. No real Qwen2.5-3B weights were available,
because the container's network policy blocks huggingface.co.

The real experiment is
`release/v0.10.0/Digital_Mirror_Recursive_v0_10.ipynb` on a Colab GPU. Its
preflight cell is the first real-model check.

## Software checks

```text
python -B -m unittest discover -s tests -v
  Python 3.11, 3.12, 3.13 (standard library only): 102 tests, OK (7 opt-in skipped)
RECURSIVE_MIRROR_TEST_DEVICE=cpu python -B -m unittest discover -s tests
  Python 3.12 + torch 2.14.1 + transformers 4.57.6: 102 tests, OK
```

| test module | tests | covers |
|---|---|---|
| test_core | 57 | the recursive engine (details below) |
| test_runtime | 22 | checkpoints, the HF backend, the CLI |
| test_tasks | 6 | `make-plan`, re-deriving every expected answer from the prompt text |
| test_experiment | 4 | replicated runs, control arm, failure isolation, pooled statistics, CLI |
| test_release | 6 | archive, notebook order and budgets; the committed release must embed exactly the current sources |
| test_hf_integration | 7 | opt-in, real library stack (details below) |

`test_core` covers:

- live-stack recursion;
- frozen proposals;
- answer-key isolation;
- one lever per proposal kind;
- the gate, including the evidence bound;
- batching;
- caching;
- budgets, including wall-clock;
- regression tests that replay each v0.8 Colab failure.

`test_runtime` covers:

- checkpoint v2/v1, tamper and limits;
- the HF backend contract: pure greedy decoding, filtered inputs, dtype;
- every CLI command.

`test_hf_integration` covers, on the real library stack:

- EOS and truncation;
- the input limit;
- **batched rows equal single requests exactly (CPU float32)**;
- a full recursive run;
- the CLI.

## What running the real stack found and fixed

- **Repetition penalty.** Qwen2.5's `repetition_penalty: 1.05` is applied
  under `do_sample=False` (confirmed: `RepetitionPenaltyLogitsProcessor` is
  active). Decoding is now plain argmax.
- `generate()` rejected `token_type_ids`. Only `input_ids` and
  `attention_mask` are passed now.
- `torch_dtype` is deprecated; the `dtype=` keyword is used where supported.
- Warnings about ignored sampling flags are gone, because the flags are
  cleared explicitly.

## End-to-end pipeline check

`python -m recursive_mirror experiment` ran with the real backend and a tiny
random model on CPU:

- 2 seeds × (self + control);
- 70 calls per run;
- batched generation;
- summary, reports and diagnostics.

The random model produces gibberish, and `diagnose` flagged exactly that:
every reply truncated, no `ANSWER:` lines, no valid forecasts. The preflight
shows the same check for the real model before the long run.

## Multi-agent review

Four specialist reviewers each had a paired skeptic:

- real inference and Colab;
- small-LLM protocol behaviour, using the v0.8 Qwen journals;
- selection statistics;
- persistence, CLI and tasks.

Confirmed or adopted findings are documented in
`docs/v0.8-trial-review.md` under "Found before the first v0.9 run".

## What this does not establish

These checks establish software behaviour, not learning. It is still untested
whether Qwen2.5-3B can improve its own strategy or self-model and transfer the
gain to held-out tasks. In particular, it is untested whether it can do better
than a hand-written step-by-step prompt. No awareness claim is made.
