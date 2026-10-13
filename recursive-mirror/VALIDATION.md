# v0.9 implementation validation (13 October 2026)

Status: the recursive self-improvement revision is implemented and passes the
local software checks. **No real-model v0.9 trial has been run yet.** The
container used for this revision has no GPU, and its network policy blocks
huggingface.co, so model weights could not be obtained. The bounded Colab T4
trial in `release/v0.9.0/Digital_Mirror_Recursive_v0_9.ipynb` is the next step.

## Software checks

```text
python -B -m unittest discover -s tests -v
Ran 80 tests ... OK            (Python 3.11, 3.12 and 3.13.16; Linux; standard library only)
```

The packaged source archive was also extracted to an empty directory and its
own test suite run there, as the notebook does. That gave 79 tests OK and 1
skipped: the skipped test loads the real v0.8 Colab checkpoint, and the
`evidence/` folder is not shipped inside the archive.

The suite has 47 recursive-core tests (including regression tests that replay
each v0.8 Colab failure), 22 runtime/checkpoint/backend/CLI tests, 6 task-plan
tests that re-derive every expected answer independently from the prompt text,
and 5 release/notebook tests. Backends were deterministic test doubles or
mocks; no neural model was loaded or downloaded.

Verified behaviours include:

- genuine nested recursion, observed from the live call stack, with the
  improved state reaching the next level;
- every proposal journalled (frozen) before any validation request; paired
  seeds and identical task text for the parent and every candidate;
- validation and final answer keys never appear in any model request.
  Discovery reference answers appear only in reflection/proposal requests,
  only for missed tasks, and not at all with `--hide-discovery-answers`;
- answer requests carry the strategy but never the self-model;
- a truncated perspective, answer or proposal is recorded and the run
  continues. v0.8 aborted on the first truncated perspective;
- a parent's malformed forecast (the v0.8 "216" case) cannot create a
  candidate calibration gain;
- one malformed candidate forecast does not veto a real accuracy gain, but
  malformed forecasts cannot hide a calibration collapse or a coverage loss;
- the paired p-value is exact (it equals the sign test for accuracy), weights
  effect size for calibration, and is deterministic;
- unchanged, empty, truncated or over-long proposals are ineligible and cost
  no validation calls;
- the attempt ledger (accepted, passed-not-selected, rejected, ineligible,
  final transfer) persists and is shown to later levels;
- deterministic-backend replay of identical requests, and none for sampling
  backends;
- checkpoint v2 round trip, tamper and type-confusion rejection, matching
  save/load limits for all three histories, v1 (v0.8) checkpoint upgrade with
  its original hashes, and loading plus continuing the real v0.8 Colab
  checkpoint;
- `make-plan` determinism, disjointness, category balance, and the shipped
  `examples/headroom.json` equal to the generator output;
- notebook archive verification, bounded and revision-pinned GPU trial
  (at most 240 calls; the shipped plan needs at most 220), and log
  preservation on interruption.

## Scripted demo and noisy simulation (software evidence only)

`python -m recursive_mirror demo` exercises every selection path:

- level 0: `repair` is selected (accuracy 0% → 100%), and `recalibrate`
  passes but is not selected;
- level 1: `repair` is unchanged, and `recalibrate` is rejected (no declared
  gain);
- the final transfer passes (accuracy 0% → 100% on 3 tasks).

The double is programmed to improve. It is not LLM evidence.

A noisy simulator, kept outside the repository, ran the full headroom plan. It
emits malformed forecasts, truncated answers and reflections, over-long and
non-executable proposals, and random answer errors. It exposed two gate
weaknesses, both fixed before release:

1. Requiring every candidate forecast to be valid let format noise veto real
   accuracy gains.
2. Accepting any single net win let a junk "expand the training dataset"
   strategy pass on noise alone.

After the fixes, runs completed with no aborts, the largest request stayed
under 8,100 characters (far below the 8,192-token input bound), and the junk
strategies were rejected or rolled back by the final test.

## What this does not establish

These checks establish software behaviour only. Whether a real 3B model can
improve its own strategy or self-model and transfer that gain to held-out
tasks is untested. No v0.9 model improvement, awareness or general
self-improvement result has been established. The v0.8 trial results remain
as recorded in `evidence/` and `docs/v0.8-trial-review.md`.
