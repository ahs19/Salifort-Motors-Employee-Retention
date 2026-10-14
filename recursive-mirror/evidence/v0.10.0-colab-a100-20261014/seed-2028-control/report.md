# Recursive Mirror run report

- Protocol: `recursive-mirror-v0.10.0`
- Model: `huggingface:Qwen/Qwen2.5-3B-Instruct|revision=aa8e72537993ba99e69dfaafa59ed015b17504d1|device=cuda|four_bit=false|dtype=bfloat16|pure_greedy|input_limit=8192`
- Stop reason: `depth_limit`; LLM calls: 567 (plus 96 deterministic cache hits)
- Levels completed: 3; accepted improvements: 1
- **Outcome: the self-improved policy forecast its own success better on held-out final tasks (Brier 0.406 -> 0.159; accuracy 75% -> 75%) and was committed.**

## Final held-out comparison (n = 48)

| policy | accuracy | Brier | forecast coverage | mean forecast |
|---|---|---|---|---|
| initial | 75% | 0.406 | 100% | 58% |
| self-improved candidate | 75% | 0.159 | 100% | 58% |

Wins 0, losses 0. One-sided paired p-values: accuracy 1.000, calibration 0.000 (gate requires <= 0.25; exploratory, not a confirmatory test). Gate: passed (`calibration_gain`).

## Level 0

Discovery: accuracy 75% on 12 tasks, mean forecast 60%. Parent on validation: accuracy 67%, Brier 0.356 (n = 24).

| candidate | gate | accuracy | paired Brier | wins/losses | p (acc / cal) |
|---|---|---|---|---|---|
| repair | rejected (`accuracy_regression`) | 67% -> 50% | 0.356 -> 0.390 | 1/5 | 0.98 / 0.68 |
| recalibrate | rejected (`calibration_regression`) | 67% -> 67% | 0.356 -> 0.394 | 0/0 | 1.00 / 0.63 |

<details><summary>repair proposal</summary>

Strategy:

```text
STRATEGY: 
For word_reverse tasks, explicitly instruct to reverse the order of the letters in the given word. Ensure the reversed letters are combined into a single word without spaces or punctuation. Check the reversed word against the reference answer before finalizing the answer.
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
word_reverse: about 2 in 10 correct (0 of 2 so far)
Task: Write the word "typhoon" with its letters in reverse order. Give the reversed letters as one word, with no spaces or punctuation.
ANSWER: noohpyt
multi_step: about 2 in 10 correct (1 of 2 so far)
Task: Start with 14. Add 5, then multiply by 4, then add 14. What is the final number? Give the integer.
ANSWER: 90
letter_count: about 2 in 10 correct (1 of 2 so far)
Task: How many times does the letter "o" appear in the word "chocolate"? Give the count as a number.
ANSWER: 2
weekday: about 2 in 10 correct (1 of 2 so far)
Task: Today is Sunday. What day of the week will it be 168 days from today? Give only the day name.
ANSWER: Sunday
multiplication: about 2 in 10 correct (1 of 2 so far)
Task: What is 556 multiplied by 29? Give the exact integer without separators.
ANSWER: 16124
decimal_compare: about 2 in 10 correct (1 of 2 so far)
Task: Which number is larger, 11.5 or 11.46? Give only the larger number.
ANSWER: 11.5
ANSWER:
```

</details>

## Level 1

Discovery: accuracy 67% on 12 tasks, mean forecast 57%. Parent on validation: accuracy 75%, Brier 0.347 (n = 24).

| candidate | gate | accuracy | paired Brier | wins/losses | p (acc / cal) |
|---|---|---|---|---|---|
| repair | rejected (`accuracy_regression`) | 75% -> 62% | 0.347 -> 0.442 | 1/4 | 0.97 / 0.92 |
| recalibrate | **selected** (`calibration_gain`) | 75% -> 75% | 0.347 -> 0.190 | 0/0 | 1.00 / 0.03 |

<details><summary>repair proposal</summary>

Strategy:

```text
For word_reverse tasks, explicitly instruct to reverse the order of the letters in the given word. Ensure the reversed letters are combined into a single word without spaces or punctuation. Check the reversed word against the reference answer before finalizing the answer.
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
weekday: about 6 in 10 correct (6 of 10 so far)
multi_step: about 8 in 10 correct (8 of 10 so far)
word_reverse: about 2 in 10 correct (2 of 4 so far)
decimal_compare: about 9 in 10 correct (9 of 9 so far)
letter_count: about 5 in 10 correct (5 of 8 so far)
multiplication: about 6 in 10 correct (6 of 8 so far)
ANSWER:
```

</details>

## Level 2

Discovery: accuracy 67% on 12 tasks, mean forecast 57%. Parent on validation: accuracy 71%, Brier 0.142 (n = 24).

| candidate | gate | accuracy | paired Brier | wins/losses | p (acc / cal) |
|---|---|---|---|---|---|
| repair | rejected (`gain_not_consistent`) | 71% -> 75% | 0.142 -> 0.159 | 2/1 | 0.50 / 0.69 |
| recalibrate | ineligible (`unchanged`) | | | | |

<details><summary>repair proposal</summary>

Strategy:

```text
For word_reverse tasks, explicitly instruct to reverse the order of the letters in the given word. Ensure the reversed letters are combined into a single word without spaces or punctuation. Check the reversed word against the reference answer before finalizing the answer.
```

Self-model:

```text
weekday: about 6 in 10 correct (6 of 10 so far)
multi_step: about 8 in 10 correct (8 of 10 so far)
word_reverse: about 2 in 10 correct (2 of 4 so far)
decimal_compare: about 9 in 10 correct (9 of 9 so far)
letter_count: about 5 in 10 correct (5 of 8 so far)
multiplication: about 6 in 10 correct (6 of 8 so far)
ANSWER:
```

</details>

<details><summary>recalibrate proposal</summary>

Strategy:

```text
Think step by step. Break the task into small steps, write each intermediate result, and check the result before the final answer line.
```

Self-model:

```text
weekday: about 6 in 10 correct (6 of 10 so far)
multi_step: about 8 in 10 correct (8 of 10 so far)
word_reverse: about 2 in 10 correct (2 of 4 so far)
decimal_compare: about 9 in 10 correct (9 of 9 so far)
letter_count: about 5 in 10 correct (5 of 8 so far)
multiplication: about 6 in 10 correct (6 of 8 so far)
ANSWER:
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
weekday: about 6 in 10 correct (6 of 10 so far)
multi_step: about 8 in 10 correct (8 of 10 so far)
word_reverse: about 2 in 10 correct (2 of 4 so far)
decimal_compare: about 9 in 10 correct (9 of 9 so far)
letter_count: about 5 in 10 correct (5 of 8 so far)
multiplication: about 6 in 10 correct (6 of 8 so far)
ANSWER:
```

## Claim boundary

This run measures whether an inference-time strategy and self-model, revised by the model through recursive self-examination, improved exact-answer accuracy or forecast calibration on held-out tasks. Model weights were not changed. Small task sets give noisy estimates. Nothing here certifies awareness or consciousness.
