"""A transparent deterministic test double; none of its scores are model evidence."""
from __future__ import annotations

import json
import re

from .core import Plan, Request, Round, Task


class ScriptedTestDouble:
    identity = "scripted-test-double"

    def complete(self, request: Request) -> str:
        context = json.loads(request.prompt)
        if request.purpose == "forecast":
            return "100" if "order-of-operations" in context["agent"]["strategy"] else "80"
        if request.purpose == "answer":
            match = re.fullmatch(r"Compute (\d+) \+ (\d+) \* (\d+)\. Return only the integer\.",
                                 context["task"]["prompt"])
            if not match:
                raise ValueError("The scripted test double only handles its documented arithmetic fixture")
            a, b, c = map(int, match.groups())
            correct_strategy = "order-of-operations" in context["agent"]["strategy"]
            return str(a + b * c if correct_strategy else (a + b) * c)
        if request.purpose.startswith("perspective:"):
            angle = request.purpose.split(":", 1)[1]
            views = {
                "actor": "From the actor angle, a single call applies a strategy; precedence mistakes can explain failures.",
                "reviewer": "From the reviewer angle, prior confidence can exceed observed accuracy. Test multiplication before addition.",
                "whole_system": "From the whole-system angle, several calls form one agent. These descriptions coexist in different contexts.",
            }
            return views[angle]
        if request.purpose == "revise_strategy":
            return "Apply order-of-operations: multiply before adding. Return only the requested integer."
        if request.purpose == "revise_self_model":
            return "This scripted arithmetic fixture is solved by multiplication before addition. Other abilities remain untested."
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
