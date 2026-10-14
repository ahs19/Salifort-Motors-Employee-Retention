# Recursive Mirror experiment summary

Model: `huggingface:Qwen/Qwen2.5-3B-Instruct|revision=aa8e72537993ba99e69dfaafa59ed015b17504d1|device=cuda|four_bit=false|dtype=bfloat16|pure_greedy|input_limit=8192`

## Runs

| seed | arm | status | levels accepted | final accuracy (initial -> selected) | wins/losses | committed |
|---|---|---|---|---|---|---|
| 2026 | control | depth_limit | 1/3 | 69% -> 69% (n=48) | 0/0 | yes |
| 2026 | self | depth_limit | 1/3 | 60% -> 77% (n=48) | 9/1 | yes |
| 2027 | control | depth_limit | 1/3 | 75% -> 75% (n=48) | 0/0 | yes |
| 2027 | self | depth_limit | 1/3 | 71% -> 71% (n=48) | 0/0 | yes |
| 2028 | control | depth_limit | 1/3 | 75% -> 75% (n=48) | 0/0 | yes |
| 2028 | self | depth_limit | 1/3 | 62% -> 62% (n=48) | 0/0 | yes |

## Pooled held-out comparisons

| comparison | tasks | accuracy | wins/losses | p (accuracy) | forecast coverage | Brier on paired valid forecasts | p (Brier) |
|---|---|---|---|---|---|---|---|
| Selected policy vs its starting policy (self arm) | 144 | 65% -> 70% | 9/1 | 0.011 | 98% -> 100% | 0.328 -> 0.198 (n=141) | 0.000 |
| Self-improved policy vs hand-written step-by-step prompt | 144 | 73% -> 70% | 9/13 | 0.857 | 100% -> 100% | 0.393 -> 0.199 (n=144) | 0.000 |
| Selected policy vs the hand-written starting prompt (control arm) | 144 | 73% -> 73% | 0/0 | 1.000 | 100% -> 100% | 0.393 -> 0.231 (n=144) | 0.000 |

## Held-out accuracy by task type (self arm, pooled)

| task type | n | initial | selected |
|---|---|---|---|
| decimal_compare | 24 | 100% | 100% |
| letter_count | 24 | 42% | 46% |
| multi_step | 24 | 100% | 100% |
| multiplication | 24 | 50% | 75% |
| weekday | 24 | 96% | 100% |
| word_reverse | 24 | 0% | 0% |

p-values are one-sided paired sign-flip tests over pooled held-out tasks (exploratory; no correction for the three comparisons). Final tasks never influenced any selection. Nothing here measures awareness.
