# Recursive Mirror run report

- Protocol: `recursive-mirror-v0.10.0`
- Model: `huggingface:Qwen/Qwen2.5-3B-Instruct|revision=aa8e72537993ba99e69dfaafa59ed015b17504d1|device=cuda|four_bit=false|dtype=bfloat16|pure_greedy|input_limit=8192`
- Stop reason: `depth_limit`; LLM calls: 639 (plus 72 deterministic cache hits)
- Levels completed: 3; accepted improvements: 1
- **Outcome: the self-improved policy answered more held-out final tasks correctly (60% -> 77%) and was committed.**

## Final held-out comparison (n = 48)

| policy | accuracy | Brier | forecast coverage | mean forecast |
|---|---|---|---|---|
| initial | 60% | 0.363 | 96% | 62% |
| self-improved candidate | 77% | 0.316 | 100% | 36% |

Wins 9, losses 1. One-sided paired p-values: accuracy 0.011, calibration 0.390 (gate requires <= 0.25; exploratory, not a confirmatory test). Gate: passed (`accuracy_gain`).

## Level 0

Discovery: accuracy 50% on 12 tasks, mean forecast 60%. Parent on validation: accuracy 58%, Brier 0.343 (n = 24).

| candidate | gate | accuracy | paired Brier | wins/losses | p (acc / cal) |
|---|---|---|---|---|---|
| repair | rejected (`gain_not_consistent`) | 58% -> 62% | 0.315 -> 0.334 | 2/1 | 0.50 / 0.59 |
| recalibrate | rejected (`no_declared_gain`) | 58% -> 58% | 0.315 -> 0.355 | 0/0 | 1.00 / 0.72 |

<details><summary>repair proposal</summary>

Strategy:

```text
Solve each task carefully. For multiplication tasks, include a detailed verification step to ensure accuracy. Adjust forecasts for multiplication tasks to 30-40% to better reflect task complexity. Before finalizing answers, verify the multiplication process.
```

Self-model:

```text
My capabilities are uncertain. Use observed outcomes to calibrate forecasts.
```

</details>

<details><summary>recalibrate proposal</summary>

Strategy:

```text
Solve each task carefully. Return only the requested answer.
```

Self-model:

```text
multiplication: about 3 of 4 correct (3 of 4 so far)
weekday: about 1 of 3 correct (1 of 3 so far)
decimal_compare: about 2 of 2 correct (2 of 2 so far)
multi_step: about 2 of 2 correct (2 of 2 so far)
letter_count: about 1 of 2 correct (1 of 2 so far)
word_reverse: about 0 of 2 correct (0 of 2 so far)
```

</details>

## Level 1

Discovery: accuracy 67% on 12 tasks, mean forecast 59%. Parent on validation: accuracy 67%, Brier 0.369 (n = 24).

| candidate | gate | accuracy | paired Brier | wins/losses | p (acc / cal) |
|---|---|---|---|---|---|
| repair | **selected** (`accuracy_gain`) | 67% -> 79% | 0.369 -> 0.352 | 4/1 | 0.19 / 0.42 |
| recalibrate | passed (`calibration_gain`) | 67% -> 67% | 0.369 -> 0.309 | 0/0 | 1.00 / 0.21 |

<details><summary>repair proposal</summary>

Strategy:

```text
Solve each task carefully. For multiplication tasks, include a detailed verification step to ensure accuracy. Adjust forecasts for multiplication tasks to 30-40% to better reflect task complexity. Before finalizing answers, verify the multiplication process. For weekday tasks, adjust forecasts to 60-70% to better reflect task complexity. Before finalizing answers, verify the day of the week calculation.
```

Self-model:

```text
My capabilities are uncertain. Use observed outcomes to calibrate forecasts.
```

</details>

<details><summary>recalibrate proposal</summary>

Strategy:

```text
Solve each task carefully. Return only the requested answer.
```

Self-model:

```text
decimal_compare: about 8 of 8 correct (8 of 8 so far)
letter_count: about 4 of 8 correct (4 of 8 so far)
multi_step: about 8 of 8 correct (8 of 8 so far)
multiplication: about 4 of 8 correct (4 of 8 so far)
weekday: about 7 of 8 correct (7 of 8 so far)
word_reverse: about 1 of 8 correct (1 of 8 so far)
```

</details>

## Level 2

Discovery: accuracy 75% on 12 tasks, mean forecast 38%. Parent on validation: accuracy 71%, Brier 0.217 (n = 24).

| candidate | gate | accuracy | paired Brier | wins/losses | p (acc / cal) |
|---|---|---|---|---|---|
| repair | rejected (`gain_not_consistent`) | 71% -> 79% | 0.217 -> 0.228 | 3/1 | 0.31 / 0.60 |
| recalibrate | rejected (`calibration_regression`) | 71% -> 71% | 0.217 -> 0.259 | 0/0 | 1.00 / 0.96 |

<details><summary>repair proposal</summary>

Strategy:

```text
Solve each task carefully. For multiplication tasks, include a detailed verification step to ensure accuracy. Adjust forecasts for multiplication tasks to 40% to better reflect task complexity. Before finalizing answers, verify the multiplication process. For weekday tasks, adjust forecasts to 70-80% to better reflect task complexity. Before finalizing answers, verify the day of the week calculation.
```

Self-model:

```text
My capabilities are uncertain. Use observed outcomes to calibrate forecasts.
```

</details>

<details><summary>recalibrate proposal</summary>

Strategy:

```text
Solve each task carefully. For multiplication tasks, include a detailed verification step to ensure accuracy. Adjust forecasts for multiplication tasks to 30-40% to better reflect task complexity. Before finalizing answers, verify the multiplication process. For weekday tasks, adjust forecasts to 60-70% to better reflect task complexity. Before finalizing answers, verify the day of the week calculation.
```

Self-model:

```text
letter_count: about 3 of 5 correct (3 of 5 so far)
multiplication: about 3 of 6 correct (3 of 6 so far)
weekday: about 6 of 6 correct (6 of 6 so far)
word_reverse: about 1 of 6 correct (1 of 6 so far)
multi_step: about 6 of 6 correct (6 of 6 so far)
```

</details>

## Initial policy

Strategy:

```text
Solve each task carefully. Return only the requested answer.
```

Self-model:

```text
My capabilities are uncertain. Use observed outcomes to calibrate forecasts.
```

## Committed policy

Strategy:

```text
Solve each task carefully. For multiplication tasks, include a detailed verification step to ensure accuracy. Adjust forecasts for multiplication tasks to 30-40% to better reflect task complexity. Before finalizing answers, verify the multiplication process. For weekday tasks, adjust forecasts to 60-70% to better reflect task complexity. Before finalizing answers, verify the day of the week calculation.
```

Self-model:

```text
My capabilities are uncertain. Use observed outcomes to calibrate forecasts.
```

## Claim boundary

This run measures whether an inference-time strategy and self-model, revised by the model through recursive self-examination, improved exact-answer accuracy or forecast calibration on held-out tasks. Model weights were not changed. Small task sets give noisy estimates. Nothing here certifies awareness or consciousness.
