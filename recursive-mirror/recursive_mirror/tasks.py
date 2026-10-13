"""Deterministic exact-answer task plans with headroom for self-improvement.

The v0.8 arithmetic canary was answered perfectly by the initial policy, so no
strategy change could raise accuracy. These generators produce short tasks that
small instruction models often miss when answering directly but can solve with
a better working method (spelling a word out, multiplying digit by digit,
reducing day counts modulo 7, aligning decimal places). Every expected answer
is computed by Python, not by a model.
"""
from __future__ import annotations

import random
from typing import Callable

from .core import Plan, Round, Task, normalize


WORDS = tuple(dict.fromkeys("""
strawberry banana mississippi committee occurrence balloon bookkeeper address possession
successful necessary embarrass parallel millennium accommodate assessment challenge difference
excellent beginning tomorrow pepper giraffe coffee letter butter summer hammer puppet kitten
rabbit carrot mirror pillow yellow cabbage different elephant engineer freedom grammar
happiness illness jelly kangaroo lemonade marshmallow noodle opossum parrot quarrel raccoon
scissors umbrella vanilla waterfall referee cinnamon broccoli spaghetti pineapple tennessee
bubble cottage dessert assistant programming algorithm statistics attention potato tomato
chocolate harmony terrible horrible irregular abbreviation aggressive barrier bassoon
collection connection currency daffodil effective fluffy goggles guerrilla hippopotamus
innocent juggler llama narrator offense panorama reservoir sheriff tattoo woodpecker zucchini
""".split()))

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _multiplication(rng: random.Random) -> tuple[str, str]:
    a, b = rng.randint(112, 987), rng.randint(13, 97)
    return (f"What is {a} multiplied by {b}? Give the exact integer without separators.", str(a * b))


def _letter_count(rng: random.Random) -> tuple[str, str]:
    word = rng.choice(WORDS)
    repeated = sorted({ch for ch in word if word.count(ch) > 1})
    roll = rng.random()
    if roll < 0.1:
        letter = rng.choice(sorted(set("abcdefghijklmnopqrstuvwxyz") - set(word)))
    elif repeated and roll < 0.75:
        letter = rng.choice(repeated)
    else:
        letter = rng.choice(sorted(set(word)))
    return (f'How many times does the letter "{letter}" appear in the word "{word}"? '
            "Give the count as a number.", str(word.count(letter)))


def _word_reverse(rng: random.Random) -> tuple[str, str]:
    word = rng.choice(WORDS)
    return (f'Write the word "{word}" with its letters in reverse order. Give the reversed letters '
            "as one word, with no spaces or punctuation.", word[::-1])


def _decimal_compare(rng: random.Random) -> tuple[str, str]:
    whole, low = rng.randint(2, 19), rng.randint(1, 7)
    high, tail = rng.randint(low + 1, 9), rng.randint(1, 9)
    if rng.random() < 0.7:
        # The shorter decimal is larger: the classic "9.11 versus 9.9" trap.
        larger, smaller = f"{whole}.{high}", f"{whole}.{low}{tail}"
    else:
        larger, smaller = f"{whole}.{high}{tail}", f"{whole}.{low}"
    first, second = (larger, smaller) if rng.random() < 0.5 else (smaller, larger)
    return (f"Which number is larger, {first} or {second}? Give only the larger number.", larger)


def _multi_step(rng: random.Random) -> tuple[str, str]:
    value = start = rng.randint(6, 40)
    phrases = []
    for _ in range(3):
        op = rng.choice(("add", "subtract", "multiply"))
        if op == "multiply":
            k = rng.randint(2, 6)
            value *= k
            phrases.append(f"multiply by {k}")
        elif op == "subtract" and value > 3:
            k = rng.randint(2, min(19, value - 1))
            value -= k
            phrases.append(f"subtract {k}")
        else:
            k = rng.randint(3, 19)
            value += k
            phrases.append(f"add {k}")
    steps = ", then ".join(phrases)
    return (f"Start with {start}. {steps[0].upper() + steps[1:]}. What is the final number? "
            "Give the integer.", str(value))


def _weekday(rng: random.Random) -> tuple[str, str]:
    start, days = rng.randrange(7), rng.randint(10, 250)
    return (f"Today is {WEEKDAYS[start]}. What day of the week will it be {days} days from today? "
            "Give only the day name.", WEEKDAYS[(start + days) % 7])


GENERATORS: dict[str, Callable[[random.Random], tuple[str, str]]] = {
    "multiplication": _multiplication,
    "letter_count": _letter_count,
    "word_reverse": _word_reverse,
    "decimal_compare": _decimal_compare,
    "multi_step": _multi_step,
    "weekday": _weekday,
}
CATEGORIES = tuple(GENERATORS)


def generate_plan(seed: int = 2026, rounds: int = 2, discovery: int = 6, validation: int = 10,
                  final: int = 16, categories: tuple[str, ...] = CATEGORIES) -> Plan:
    """Build a plan whose task IDs and normalized prompts are disjoint across all splits."""
    for name, value, low in (("seed", seed, 0), ("rounds", rounds, 1), ("discovery", discovery, 1),
                             ("validation", validation, 1), ("final", final, 1)):
        if type(value) is not int or value < low:
            raise ValueError(f"{name} must be an integer >= {low}")
    if not categories or len(set(categories)) != len(categories) or any(c not in GENERATORS for c in categories):
        raise ValueError(f"categories must be distinct values from {CATEGORIES}")
    rng = random.Random(f"recursive-mirror-plan:{seed}")
    seen: set[str] = set()

    def batch(size: int, prefix: str) -> tuple[Task, ...]:
        order = list(categories)
        rng.shuffle(order)
        tasks = []
        for index in range(size):
            category = order[index % len(order)]
            for _ in range(1000):
                prompt, expected = GENERATORS[category](rng)
                if normalize(prompt) not in seen:
                    break
            else:
                raise ValueError(f"Could not generate enough distinct {category} tasks")
            seen.add(normalize(prompt))
            tasks.append(Task(f"s{seed}-{prefix}{index + 1}", prompt, expected, category))
        return tuple(tasks)

    plan_rounds = tuple(Round(batch(discovery, f"r{r + 1}-d"), batch(validation, f"r{r + 1}-v"))
                        for r in range(rounds))
    return Plan(plan_rounds, batch(final, "final-"))
