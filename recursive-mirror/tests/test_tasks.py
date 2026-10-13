"""The generated plans must be deterministic, disjoint and correctly keyed."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
import unittest

from recursive_mirror.core import normalize
from recursive_mirror.runtime import load_plan
from recursive_mirror.tasks import CATEGORIES, WEEKDAYS, generate_plan


ROOT = Path(__file__).resolve().parents[1]


def solve(category: str, prompt: str) -> str:
    """An independent re-derivation of each expected answer from the prompt text."""
    if category == "multiplication":
        a, b = map(int, re.fullmatch(r"What is (\d+) multiplied by (\d+)\? .*", prompt).groups())
        return str(a * b)
    if category == "letter_count":
        letter, word = re.fullmatch(r'How many times does the letter "(\w)" appear in the word "(\w+)"\? .*',
                                    prompt).groups()
        return str(sum(ch == letter for ch in word))
    if category == "word_reverse":
        return "".join(reversed(re.fullmatch(r'Write the word "(\w+)" .*', prompt).group(1)))
    if category == "decimal_compare":
        x, y = re.fullmatch(r"Which number is larger, ([\d.]+) or ([\d.]+)\? .*", prompt).groups()
        return x if float(x) > float(y) else y
    if category == "multi_step":
        value = int(re.match(r"Start with (\d+)\.", prompt).group(1))
        for op, k in re.findall(r"(add|subtract|multiply by) (\d+)", prompt, re.IGNORECASE):
            op, k = op.lower(), int(k)
            value = value + k if op == "add" else value - k if op == "subtract" else value * k
        return str(value)
    if category == "weekday":
        day, days = re.fullmatch(r"Today is (\w+)\. What day of the week will it be (\d+) days from today\? .*",
                                 prompt).groups()
        return WEEKDAYS[(WEEKDAYS.index(day) + int(days)) % 7]
    raise AssertionError(category)


class TaskPlanTests(unittest.TestCase):
    def test_generation_is_deterministic_per_seed(self):
        self.assertEqual(generate_plan(7), generate_plan(7))
        self.assertNotEqual(generate_plan(7), generate_plan(8))

    def test_every_expected_answer_is_independently_correct(self):
        for seed in range(25):
            plan = generate_plan(seed, rounds=3, discovery=6, validation=10, final=20)
            for task in plan.tasks():
                with self.subTest(seed=seed, task=task.task_id):
                    self.assertEqual(solve(task.category, task.prompt), task.expected)

    def test_splits_are_disjoint_balanced_and_sized(self):
        plan = generate_plan(3, rounds=2, discovery=6, validation=8, final=12)
        self.assertEqual([len(r.discovery) for r in plan.rounds], [6, 6])
        self.assertEqual([len(r.validation) for r in plan.rounds], [8, 8])
        self.assertEqual(len(plan.final_tasks), 12)
        tasks = plan.tasks()
        self.assertEqual(len({t.task_id for t in tasks}), len(tasks))
        self.assertEqual(len({normalize(t.prompt) for t in tasks}), len(tasks))
        for batch in (plan.rounds[0].discovery, plan.final_tasks):
            self.assertEqual({t.category for t in batch}, set(CATEGORIES))
        self.assertTrue(all(t.task_id.startswith("s3-") for t in tasks))

    def test_decimal_traps_include_shorter_larger_numbers(self):
        plan = generate_plan(11, rounds=3, discovery=12, validation=12, final=24,
                             categories=("decimal_compare",))
        shorter_wins = 0
        for task in plan.tasks():
            numbers = re.findall(r"\d+\.\d+", task.prompt)
            other = next(n for n in numbers if n != task.expected)
            shorter_wins += len(task.expected) < len(other)
        self.assertGreater(shorter_wins, len(plan.tasks()) // 2)

    def test_category_subset_and_invalid_arguments(self):
        plan = generate_plan(1, categories=("weekday", "multiplication"))
        self.assertEqual({t.category for t in plan.tasks()}, {"weekday", "multiplication"})
        for kwargs in ({"rounds": 0}, {"final": 0}, {"seed": -1}, {"categories": ()},
                       {"categories": ("weekday", "weekday")}, {"categories": ("poetry",)}, {"rounds": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate_plan(**kwargs)

    def test_shipped_headroom_plan_matches_the_generator(self):
        shipped = load_plan(ROOT / "examples" / "headroom.json")
        self.assertEqual(shipped, generate_plan(2026, rounds=2, discovery=6, validation=10, final=16))
        self.assertEqual(json.loads((ROOT / "examples" / "headroom.json").read_text(encoding="utf-8")),
                         json.loads(json.dumps(asdict(shipped))))


if __name__ == "__main__":
    unittest.main()
