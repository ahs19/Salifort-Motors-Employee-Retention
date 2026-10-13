"""A transparent deterministic test double; none of its scores are model evidence."""
from __future__ import annotations

import re

from .core import Plan, Request, Round, Task, sections


class ScriptedTestDouble:
    """Programmed to improve so that every selection path is exercised.

    The default strategy evaluates left to right (wrong on these precedence
    tasks). A "repair" proposal supplies order-of-operations (right), and a
    "recalibrate" proposal only states reliability. Forecasts are read from the
    self-model's "about N%" phrase. This is plumbing, not learning.
    """
    identity = "scripted-test-double"
    deterministic = True
    FIXED = "Apply order-of-operations: multiply before adding. Check the product, then add."

    def complete(self, request: Request) -> str:
        parts = sections(request.prompt)
        if request.purpose == "forecast":
            stated = re.search(r"about (\d+)%", parts["SELF-MODEL"])
            return stated.group(1) if stated else "80"
        if request.purpose == "answer":
            match = re.search(r"Compute (\d+) \+ (\d+) \* (\d+)\. Return only the integer\.", parts["TASK"])
            if not match:
                raise ValueError("The scripted test double only handles its documented arithmetic fixture")
            a, b, c = map(int, match.groups())
            value = a + b * c if "order-of-operations" in parts["STRATEGY"] else (a + b) * c
            return f"Working through the expression.\nANSWER: {value}"
        if request.purpose.startswith("perspective:"):
            angle = request.purpose.split(":", 1)[1]
            views = {
                "actor": "From the actor angle, I added before multiplying; precedence mistakes explain the failures.",
                "reviewer": "From the reviewer angle, forecasts of 80% exceeded observed accuracy. Test multiplication first.",
                "whole_system": "From the whole-system angle, the actor and reviewer describe one agent in different contexts.",
            }
            return views[angle]
        if request.purpose.startswith("revise_strategy:"):
            return self.FIXED
        if request.purpose.startswith("revise_self_model:"):
            strategy = parts["STRATEGY THE AGENT WILL USE"]
            if request.purpose.endswith(":recalibrate"):
                if "order-of-operations" in strategy:
                    return "Precedence tasks: reliable, about 99% with this strategy."
                return "Precedence tasks: unreliable, about 10% with this strategy."
            return "Precedence tasks: about 95% reliable when multiplying before adding. Other abilities untested."
        raise ValueError(f"Unsupported scripted purpose: {request.purpose}")


def demo_plan() -> Plan:
    def task(name: str, a: int, b: int, c: int) -> Task:
        return Task(name, f"Compute {a} + {b} * {c}. Return only the integer.", str(a + b * c), "precedence")

    return Plan((
        Round((task("discover-1", 2, 3, 4),),
              (task("validate-1a", 3, 4, 5), task("validate-1b", 4, 5, 6))),
        Round((task("discover-2", 5, 6, 7),),
              (task("validate-2a", 6, 7, 8), task("validate-2b", 7, 8, 9))),
    ), (task("final-a", 8, 9, 3), task("final-b", 9, 3, 4), task("final-c", 4, 8, 2)))
