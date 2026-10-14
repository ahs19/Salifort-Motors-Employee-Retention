# Recursive Mirror run report

- Protocol: `recursive-mirror-v0.10.0`
- Model: `huggingface:Qwen/Qwen2.5-3B-Instruct|revision=aa8e72537993ba99e69dfaafa59ed015b17504d1|device=cuda|four_bit=false|dtype=bfloat16|pure_greedy|input_limit=8192`
- Stop reason: `depth_limit`; LLM calls: 567 (plus 96 deterministic cache hits)
- Levels completed: 3; accepted improvements: 1
- **Outcome: the self-improved policy forecast its own success better on held-out final tasks (Brier 0.323 -> 0.124; accuracy 62% -> 62%) and was committed.**

## Final held-out comparison (n = 48)

| policy | accuracy | Brier | forecast coverage | mean forecast |
|---|---|---|---|---|
| initial | 62% | 0.323 | 100% | 66% |
| self-improved candidate | 62% | 0.124 | 100% | 57% |

Wins 0, losses 0. One-sided paired p-values: accuracy 1.000, calibration 0.000 (gate requires <= 0.25; exploratory, not a confirmatory test). Gate: passed (`calibration_gain`).

## Level 0

Discovery: accuracy 50% on 12 tasks, mean forecast 60%. Parent on validation: accuracy 67%, Brier 0.262 (n = 24).

| candidate | gate | accuracy | paired Brier | wins/losses | p (acc / cal) |
|---|---|---|---|---|---|
| repair | rejected (`accuracy_regression`) | 67% -> 62% | 0.262 -> 0.375 | 2/3 | 0.81 / 0.86 |
| recalibrate | rejected (`calibration_regression`) | 67% -> 67% | 0.262 -> 0.266 | 0/0 | 1.00 / 0.54 |

<details><summary>repair proposal</summary>

Strategy:

```text
For word_reverse tasks, reverse the letters directly without additional steps. Explicitly instruct: "Reverse the order of the letters in the given word without any additional operations." Check: "Reverse the letters of 'typhoon' and verify the answer matches the reference." For letter_count and multiplication, adjust forecasts to 100% to ensure correct answers. Check: "Count the occurrences of 'f' in 'difference' and multiply 251 by 93, then verify the answers."
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
word_reverse: about 100% correct (2 of 2 so far)
letter_count: about 50% correct (0 of 2 so far)
multi_step: about 50% correct (2 of 2 so far)
weekday: about 75% correct (2 of 2 so far)
multiplication: about 25% correct (0 of 2 so far)
decimal_compare: about 75% correct (2 of 2 so far)
```

</details>

## Level 1

Discovery: accuracy 67% on 12 tasks, mean forecast 64%. Parent on validation: accuracy 54%, Brier 0.320 (n = 24).

| candidate | gate | accuracy | paired Brier | wins/losses | p (acc / cal) |
|---|---|---|---|---|---|
| repair | rejected (`no_declared_gain`) | 54% -> 54% | 0.320 -> 0.397 | 2/2 | 0.69 / 0.88 |
| recalibrate | **selected** (`calibration_gain`) | 54% -> 54% | 0.320 -> 0.116 | 0/0 | 1.00 / 0.00 |

<details><summary>repair proposal</summary>

Strategy:

```text
Solve each task carefully. Return only the requested answer.

For word_reverse tasks, reverse the order of the letters directly without additional steps. Explicitly instruct: "Reverse the order of the letters in the given word to ensure correct answers. Check: 'Count the occurrences of 'f' in 'difference' and multiply 251 by 93, then verify the answers.'"

For weekday tasks, explicitly calculate the day of the week. Explicitly instruct: "Calculate the day of the week by determining the remainder when the number of days is divided by 7, then add the remainder to the starting day. Check: 'Today is Saturday. What day of the week will it be 250 days from today? Give only the day name.'"

ANSWER: <answer>
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
weekday: about 80% correct (7 of 8 so far)
word_reverse: about 20% correct (1 of 8 so far)
multi_step: about 90% correct (8 of 8 so far)
multiplication: about 30% correct (2 of 8 so far)
letter_count: about 50% correct (4 of 8 so far)
decimal_compare: about 90% correct (8 of 8 so far)
```

</details>

## Level 2

Discovery: accuracy 58% on 12 tasks, mean forecast 55%. Parent on validation: accuracy 62%, Brier 0.117 (n = 24).

| candidate | gate | accuracy | paired Brier | wins/losses | p (acc / cal) |
|---|---|---|---|---|---|
| repair | rejected (`accuracy_regression`) | 62% -> 33% | 0.117 -> 0.181 | 2/9 | 0.99 / 1.00 |
| recalibrate | ineligible (`self_model_truncated`) | | | | |

<details><summary>repair proposal</summary>

Strategy:

```text
Solve each task carefully. Return only the requested answer.

For word_reverse tasks, reverse the order of the letters directly without additional steps. Explicitly instruct: "Reverse the order of the letters in the given word to form the reversed word." Check: "Count the occurrences of 'f' in 'difference' and multiply 251 by 93, then verify the answers."

For letter_count tasks, explicitly count the occurrences of the letter in the given word. Check: "Count the occurrences of 'f' in 'difference' and verify the answer."

For multiplication tasks, perform the multiplication calculation directly. Check: "Calculate 223 multiplied by 32 and verify the answer."

ANSWER: <answer>
```

Self-model:

```text
weekday: about 80% correct (7 of 8 so far)
word_reverse: about 20% correct (1 of 8 so far)
multi_step: about 90% correct (8 of 8 so far)
multiplication: about 30% correct (2 of 8 so far)
letter_count: about 50% correct (4 of 8 so far)
decimal_compare: about 90% correct (8 of 8 so far)
```

</details>

<details><summary>recalibrate proposal</summary>

Strategy:

```text
Solve each task carefully. Return only the requested answer.
```

Self-model:

```text
weekday: about 80% correct (7 of 9 so far)
word_reverse: about 20% correct (1 of 9 so far)
multi_step: about 90% correct (8 of 9 so far)
multiplication: about 30% correct (2 of 9 so far)
letter_count: about 50% correct (4 of 9 so far)
decimal_compare: about 90% correct (8 of 9 so far)

=== DISCOVERY RESULTS THIS LEVEL ===
Score: 7/12 correct; mean valid forecast 55% versus accuracy 58%.
[1] multiplication | forecast 30% | INCORRECT
Task: What is 831 multiplied by 78? Give the exact integer without separators.
Agent's final answer: 64958 [answer_line]
Reference answer: 64818
Agent's reply (excerpt): To solve 831 multiplied by 78, I will perform the multiplication calculation.
831 * 78 = 64958
ANSWER: 64958
[2] letter_count | forecast 50% | INCORRECT
Task: How many times does the letter "f" appear in the word "fluffy"? Give the count as a number.
Agent's final answer: 2 [answer_line]
Reference answer: 3
Agent's reply (excerpt): To solve this task, I will count the occurrences of the letter "f" in the word "fluffy".

1 (f) 
2 (ff) 
3 (fl) 
4 (flu) 
5 (fluf) 
6 (fluffy)

The letter "f" appears 2 times in the word "fluffy".

ANSWER: 2
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
Solve each task carefully. Return only the requested answer.
```

Self-model:

```text
weekday: about 80% correct (7 of 8 so far)
word_reverse: about 20% correct (1 of 8 so far)
multi_step: about 90% correct (8 of 8 so far)
multiplication: about 30% correct (2 of 8 so far)
letter_count: about 50% correct (4 of 8 so far)
decimal_compare: about 90% correct (8 of 8 so far)
```

## Claim boundary

This run measures whether an inference-time strategy and self-model, revised by the model through recursive self-examination, improved exact-answer accuracy or forecast calibration on held-out tasks. Model weights were not changed. Small task sets give noisy estimates. Nothing here certifies awareness or consciousness.
