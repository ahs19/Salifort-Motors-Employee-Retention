# Recursive Mirror run report

- Protocol: `recursive-mirror-v0.10.0`
- Model: `huggingface:Qwen/Qwen2.5-3B-Instruct|revision=aa8e72537993ba99e69dfaafa59ed015b17504d1|device=cuda|four_bit=false|dtype=bfloat16|pure_greedy|input_limit=8192`
- Stop reason: `depth_limit`; LLM calls: 567 (plus 96 deterministic cache hits)
- Levels completed: 3; accepted improvements: 1
- **Outcome: the self-improved policy forecast its own success better on held-out final tasks (Brier 0.380 -> 0.244; accuracy 69% -> 69%) and was committed.**

## Final held-out comparison (n = 48)

| policy | accuracy | Brier | forecast coverage | mean forecast |
|---|---|---|---|---|
| initial | 69% | 0.380 | 100% | 57% |
| self-improved candidate | 69% | 0.244 | 100% | 51% |

Wins 0, losses 0. One-sided paired p-values: accuracy 1.000, calibration 0.007 (gate requires <= 0.25; exploratory, not a confirmatory test). Gate: passed (`calibration_gain`).

## Level 0

Discovery: accuracy 75% on 12 tasks, mean forecast 53%. Parent on validation: accuracy 67%, Brier 0.336 (n = 24).

| candidate | gate | accuracy | paired Brier | wins/losses | p (acc / cal) |
|---|---|---|---|---|---|
| repair | rejected (`accuracy_regression`) | 67% -> 29% | 0.336 -> 0.496 | 0/9 | 1.00 / 0.95 |
| recalibrate | **selected** (`calibration_gain`) | 67% -> 67% | 0.336 -> 0.254 | 0/0 | 1.00 / 0.15 |

<details><summary>repair proposal</summary>

Strategy:

```text
STRATEGY: 
For letter_count, ensure case sensitivity in the word examination. For word_reverse, refine the strategy to reduce overconfidence in forecasts. 
[5] letter_count: 
1. Write down the word "quizzical" to examine it closely.
2. Count how many times "i" appears in the word, considering case sensitivity.
3. Provide the count as the answer.
Check: Count the "i"s in "quizzical" and ensure it matches the reference answer.
[12] word_reverse: 
1. Identify the letters in the word "zucchini".
2. Reverse the order of these letters.
3. Combine the reversed letters into a single word.
4. Provide the final reversed word as the answer.
Check: Reverse the word "zucchini" and ensure the final answer matches the reference answer.
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
decimal_compare: about 3 of 4 correct (3 of 4 so far)
letter_count: about 1 of 3 correct (1 of 3 so far)
multi_step: about 2 of 2 correct (2 of 2 so far)
multiplication: about 2 of 2 correct (2 of 2 so far)
weekday: about 2 of 2 correct (2 of 2 so far)
word_reverse: about 0 of 2 correct (0 of 2 so far)
```

</details>

## Level 1

Discovery: accuracy 75% on 12 tasks, mean forecast 50%. Parent on validation: accuracy 79%, Brier 0.283 (n = 24).

| candidate | gate | accuracy | paired Brier | wins/losses | p (acc / cal) |
|---|---|---|---|---|---|
| repair | rejected (`accuracy_regression`) | 79% -> 46% | 0.283 -> 0.253 | 0/8 | 1.00 / 0.31 |
| recalibrate | rejected (`no_declared_gain`) | 79% -> 79% | 0.283 -> 0.283 | 0/0 | 1.00 / 0.50 |

<details><summary>repair proposal</summary>

Strategy:

```text
STRATEGY: 
For letter_count, ensure case sensitivity in the word examination. For word_reverse, refine the strategy to reduce overconfidence in forecasts. Check: Reverse the word "zucchini" and ensure the final answer matches the reference answer.

[4] letter_count | forecast 30% | INCORRECT
Task: How many times does the letter "t" appear in the word "cottage"? Give the count as a number.
Agent's final answer: 4 [answer_line]
Check: echo "cottage" | grep -o "t" | wc -l
ANSWER: 2

[11] word_reverse | forecast 50% | INCORRECT
Task: Write the word "statistics" with its letters in reverse order. Give the reversed letters as one word, with no spaces or punctuation.
Agent's final answer: statisticats [answer_line]
Check: echo "statistics" | rev | tr -d '\n' | sed 's/ //g'
ANSWER: scitsitats
```

Self-model:

```text
decimal_compare: about 3 of 4 correct (3 of 4 so far)
letter_count: about 1 of 3 correct (1 of 3 so far)
multi_step: about 2 of 2 correct (2 of 2 so far)
multiplication: about 2 of 2 correct (2 of 2 so far)
weekday: about 2 of 2 correct (2 of 2 so far)
word_reverse: about 0 of 2 correct (0 of 2 so far)
```

</details>

<details><summary>recalibrate proposal</summary>

Strategy:

```text
Think step by step. Break the task into small steps, write each intermediate result, and check the result before the final answer line.
```

Self-model:

```text
decimal_compare: about 3 of 4 correct (3 of 4 so far)
letter_count: about 1 of 3 correct (1 of 3 so far)
multi_step: about 2 of 2 correct (2 of 2 so far)
multiplication: about 2 of 2 correct (2 of 2 so far)
weekday: about 2 of 2 correct (2 of 2 so far)
word_reverse: about 0 of 2 correct (0 of 2 so far)
Check: Reverse the word "zucchini" and ensure the final answer matches the reference answer.
```

</details>

## Level 2

Discovery: accuracy 83% on 12 tasks, mean forecast 50%. Parent on validation: accuracy 83%, Brier 0.250 (n = 24).

| candidate | gate | accuracy | paired Brier | wins/losses | p (acc / cal) |
|---|---|---|---|---|---|
| repair | rejected (`accuracy_regression`) | 83% -> 54% | 0.250 -> 0.490 | 1/8 | 1.00 / 0.98 |
| recalibrate | ineligible (`unchanged`) | | | | |

<details><summary>repair proposal</summary>

Strategy:

```text
STRATEGY: 
For word_reverse, ensure case sensitivity in the word examination. For letter_count, refine the strategy to reduce underconfidence in forecasts. Implement a check to verify the final reversed word matches the reference answer.

Task: Write the word "glossary" with its letters in reverse order. Give the reversed letters as one word, with no spaces or punctuation.
Step 1: Identify the letters in the word "glossary".
Step 2: Reverse the order of these letters.
Step 3: Combine the reversed letters into a single word.
Step 4: Ensure the final answer matches the reference answer "yrassolg".
Check: echo "glossary" | rev | tr -d '\n' | sed 's/ //g' | grep -q "yrassolg"
ANSWER: yrassolg
```

Self-model:

```text
decimal_compare: about 3 of 4 correct (3 of 4 so far)
letter_count: about 1 of 3 correct (1 of 3 so far)
multi_step: about 2 of 2 correct (2 of 2 so far)
multiplication: about 2 of 2 correct (2 of 2 so far)
weekday: about 2 of 2 correct (2 of 2 so far)
word_reverse: about 0 of 2 correct (0 of 2 so far)
```

</details>

<details><summary>recalibrate proposal</summary>

Strategy:

```text
Think step by step. Break the task into small steps, write each intermediate result, and check the result before the final answer line.
```

Self-model:

```text
decimal_compare: about 3 of 4 correct (3 of 4 so far)
letter_count: about 1 of 3 correct (1 of 3 so far)
multi_step: about 2 of 2 correct (2 of 2 so far)
multiplication: about 2 of 2 correct (2 of 2 so far)
weekday: about 2 of 2 correct (2 of 2 so far)
word_reverse: about 0 of 2 correct (0 of 2 so far)
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
decimal_compare: about 3 of 4 correct (3 of 4 so far)
letter_count: about 1 of 3 correct (1 of 3 so far)
multi_step: about 2 of 2 correct (2 of 2 so far)
multiplication: about 2 of 2 correct (2 of 2 so far)
weekday: about 2 of 2 correct (2 of 2 so far)
word_reverse: about 0 of 2 correct (0 of 2 so far)
```

## Claim boundary

This run measures whether an inference-time strategy and self-model, revised by the model through recursive self-examination, improved exact-answer accuracy or forecast calibration on held-out tasks. Model weights were not changed. Small task sets give noisy estimates. Nothing here certifies awareness or consciousness.
