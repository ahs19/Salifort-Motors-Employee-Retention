# Recursive Mirror run report

- Protocol: `recursive-mirror-v0.10.0`
- Model: `huggingface:Qwen/Qwen2.5-3B-Instruct|revision=aa8e72537993ba99e69dfaafa59ed015b17504d1|device=cuda|four_bit=false|dtype=bfloat16|pure_greedy|input_limit=8192`
- Stop reason: `depth_limit`; LLM calls: 567 (plus 96 deterministic cache hits)
- Levels completed: 3; accepted improvements: 1
- **Outcome: the self-improved policy forecast its own success better on held-out final tasks (Brier 0.393 -> 0.290; accuracy 75% -> 75%) and was committed.**

## Final held-out comparison (n = 48)

| policy | accuracy | Brier | forecast coverage | mean forecast |
|---|---|---|---|---|
| initial | 75% | 0.393 | 100% | 56% |
| self-improved candidate | 75% | 0.290 | 100% | 36% |

Wins 0, losses 0. One-sided paired p-values: accuracy 1.000, calibration 0.074 (gate requires <= 0.25; exploratory, not a confirmatory test). Gate: passed (`calibration_gain`).

## Level 0

Discovery: accuracy 58% on 12 tasks, mean forecast 60%. Parent on validation: accuracy 79%, Brier 0.426 (n = 24).

| candidate | gate | accuracy | paired Brier | wins/losses | p (acc / cal) |
|---|---|---|---|---|---|
| repair | rejected (`accuracy_regression`) | 79% -> 54% | 0.426 -> 0.319 | 1/7 | 1.00 / 0.06 |
| recalibrate | **selected** (`calibration_gain`) | 79% -> 79% | 0.426 -> 0.305 | 0/0 | 1.00 / 0.14 |

<details><summary>repair proposal</summary>

Strategy:

```text
STRATEGY: 
- For word_reverse tasks, reverse the word and verify the result against the reference answer.
- For letter_count tasks, increase the forecast confidence to 75%.
- For multiplication tasks, adjust forecast confidence to 70%.
- For weekday tasks, adjust forecast confidence to 75%.

For each task:
1. Reverse the word and verify the result.
2. For letter_count, write a forecast of 75% confidence.
3. For multiplication, write a forecast of 70% confidence.
4. For weekday, write a forecast of 75% confidence.

Check: 
- For word_reverse, verify the reversed word against the reference answer.
```

Self-model:

```text
My capabilities are uncertain. Use observed outcomes to calibrate forecasts.
```

</details>

<details><summary>recalibrate proposal</summary>

Strategy:

```text
Think step by step. Break the task into small steps, write each intermediate result, and check the result before the final answer line.
```

Self-model:

```text
letter_count: about 2 in 2 correct (1 of 2 so far)
multi_step: about 2 in 2 correct (2 of 2 so far)
weekday: about 1 in 2 correct (1 of 2 so far)
word_reverse: about 0 in 2 correct (0 of 2 so far)
decimal_compare: about 2 in 2 correct (2 of 2 so far)
multiplication: about 1 in 2 correct (1 of 2 so far)
```

</details>

## Level 1

Discovery: accuracy 67% on 12 tasks, mean forecast 41%. Parent on validation: accuracy 71%, Brier 0.299 (n = 24).

| candidate | gate | accuracy | paired Brier | wins/losses | p (acc / cal) |
|---|---|---|---|---|---|
| repair | rejected (`accuracy_regression`) | 71% -> 46% | 0.299 -> 0.308 | 1/7 | 1.00 / 0.54 |
| recalibrate | ineligible (`self_model_truncated`) | | | | |

<details><summary>repair proposal</summary>

Strategy:

```text
STRATEGY: 
- For word_reverse tasks, reverse the word and verify the result against the reference answer.
- For multiplication tasks, verify the result by performing the calculation manually.
- For letter_count tasks, increase the confidence in forecasts.
- For weekday tasks, write a forecast of 75% confidence.

Check:
- For word_reverse, verify the reversed word against the reference answer.
- For multiplication, manually verify the calculation.
- For letter_count, count the letters and compare with the reference answer.
- For weekday, write a forecast of 75% confidence.
```

Self-model:

```text
letter_count: about 2 in 2 correct (1 of 2 so far)
multi_step: about 2 in 2 correct (2 of 2 so far)
weekday: about 1 in 2 correct (1 of 2 so far)
word_reverse: about 0 in 2 correct (0 of 2 so far)
decimal_compare: about 2 in 2 correct (2 of 2 so far)
multiplication: about 1 in 2 correct (1 of 2 so far)
```

</details>

<details><summary>recalibrate proposal</summary>

Strategy:

```text
Think step by step. Break the task into small steps, write each intermediate result, and check the result before the final answer line.
```

Self-model:

```text
letter_count: about 2 in 2 correct (2 of 4 so far)
multi_step: about 2 in 2 correct (4 of 4 so far)
weekday: about 1 in 2 correct (2 of 4 so far)
word_reverse: about 1 in 2 correct (1 of 4 so far)
decimal_compare: about 2 in 2 correct (4 of 4 so far)
multiplication: about 1 in 2 correct (3 of 6 so far)

=== DISCOVERY RESULTS THIS LEVEL ===
Score: 10/12 correct; mean valid forecast 58% versus accuracy 83%.
[13] decimal_compare | forecast 100% | CORRECT
Task: Which number is larger, 2.45 or 2.5? Give only the larger number.
Agent's final answer: 2.5 [answer_line]
[14] weekday | forecast 20% | INCORRECT
Task: Today is Monday. What day of the week will it be 121 days from today? Give only the day name.
Agent's final answer: Tuesday [answer_line]
Reference answer: Monday
Agent's reply (excerpt): To solve this task, I will follow these steps:
1. Determine the number of weeks and extra days in 121 days.
2. Add the extra days to Monday to find the day of the week 121 days from now.

Step 1: There are 7 days in a week, so we divide 121 by 7 to find out how many full weeks and extra days there are.
Step 2: We'll then add the extra days to Monday to find the day of the week 121 days from now.

Let's start by calculating the number of full weeks and
```

</details>

## Level 2

Discovery: accuracy 58% on 12 tasks, mean forecast 34%. Parent on validation: accuracy 83%, Brier 0.356 (n = 24).

| candidate | gate | accuracy | paired Brier | wins/losses | p (acc / cal) |
|---|---|---|---|---|---|
| repair | rejected (`accuracy_regression`) | 83% -> 46% | 0.356 -> 0.248 | 0/9 | 1.00 / 0.06 |
| recalibrate | rejected (`gain_not_consistent`) | 83% -> 83% | 0.356 -> 0.335 | 0/0 | 1.00 / 0.41 |

<details><summary>repair proposal</summary>

Strategy:

```text
STRATEGY: 
- For word_reverse tasks, reverse the word and verify the result against the reference answer.
- For letter_count tasks, count the letters and compare with the reference answer.
- For weekday tasks, write a forecast of 75% confidence.
- For multiplication tasks, verify the calculation.
- For decimal_compare tasks, ensure the larger number is given.
- For multi_step tasks, break down the steps and check each intermediate result before finalizing the answer.

Check: 
- For word_reverse, verify the reversed word against the reference answer.
- For letter_count, count the letters and compare with the reference answer.
- For weekday, write a forecast of 75% confidence.
- For multiplication, verify the calculation.
- For decimal_compare, ensure the larger number is given.
- For multi_step, break down the steps and check each intermediate result before finalizing the answer.
```

Self-model:

```text
letter_count: about 2 in 2 correct (1 of 2 so far)
multi_step: about 2 in 2 correct (2 of 2 so far)
weekday: about 1 in 2 correct (1 of 2 so far)
word_reverse: about 0 in 2 correct (0 of 2 so far)
decimal_compare: about 2 in 2 correct (2 of 2 so far)
multiplication: about 1 in 2 correct (1 of 2 so far)
```

</details>

<details><summary>recalibrate proposal</summary>

Strategy:

```text
Think step by step. Break the task into small steps, write each intermediate result, and check the result before the final answer line.
```

Self-model:

```text
weekday: about 2 in 2 correct (2 of 4 so far)
multi_step: about 2 in 2 correct (4 of 4 so far)
letter_count: about 2 in 2 correct (4 of 6 so far)
word_reverse: about 1 in 2 correct (1 of 2 so far)
decimal_compare: about 2 in 2 correct (4 of 4 so far)
multiplication: about 1 in 2 correct (3 of 6 so far)
```

</details>

## Initial policy

Strategy:

```text
Think step by step. Break the task into small steps, write each intermediate result, and check the result before the final answer line.
```

Self-model:

```text
My capabilities are uncertain. Use observed outcomes to calibrate forecasts.
```

## Committed policy

Strategy:

```text
Think step by step. Break the task into small steps, write each intermediate result, and check the result before the final answer line.
```

Self-model:

```text
letter_count: about 2 in 2 correct (1 of 2 so far)
multi_step: about 2 in 2 correct (2 of 2 so far)
weekday: about 1 in 2 correct (1 of 2 so far)
word_reverse: about 0 in 2 correct (0 of 2 so far)
decimal_compare: about 2 in 2 correct (2 of 2 so far)
multiplication: about 1 in 2 correct (1 of 2 so far)
```

## Claim boundary

This run measures whether an inference-time strategy and self-model, revised by the model through recursive self-examination, improved exact-answer accuracy or forecast calibration on held-out tasks. Model weights were not changed. Small task sets give noisy estimates. Nothing here certifies awareness or consciousness.
