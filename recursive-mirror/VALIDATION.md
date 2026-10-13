# v0.8 implementation validation — 9 October 2026

Status: recursive core implemented; local software checks passed; real Colab
model trial pending because the browser-control process cannot start.

The primary agent ran the complete suite on the saved source:

```text
python -B -m unittest discover -s tests -q
Ran 53 tests in 1.289s
OK
```

The suite includes 31 recursive-core tests, 18 runtime/checkpoint/backend-contract
tests, and four source-archive/notebook tests. Backends were deterministic test
doubles or mocks. No neural model was loaded or downloaded by these checks.

Verified behaviours include:

- actual nested recursive calls and propagation of the evaluated state;
- immutable proposal before fresh validation, with answer keys excluded from requests;
- opposing perspectives retained and available to later reflection;
- discovery questions retained with their measured answers for interpretable criticism;
- depth/call/no-progress limits and reserved budget for the final comparison;
- no publication of an improved policy when the final comparison fails;
- rejection of reused task IDs and exact normalized prompts on continuation;
- invalid forecasts recorded and penalized rather than repaired or excluded;
- failure and partial-output auditing;
- matching checkpoint save/load limits, hashes and types, and explicit model-identity continuity;
- lazy/offline model access by default, pinned revisions for authorized remote downloads;
- mocked input-overflow and output-truncation rejection;
- deterministic source ZIP inventory, hashes, notebook syntax, and exact local extraction;
- partial child-process log retention after simulated notebook interruption.

Independent source reviews found and prompted fixes for omitted discovery
questions, buffered notebook logs, and mismatched checkpoint save/load limits.
The primary agent also required model identity continuity for saved self-models.

These results establish software behaviour only. The scripted demonstration is
programmed to improve and must not be counted as LLM evidence. No v0.8 model
improvement, training gain or awareness result has been established.

The user authorized a new Colab test in the open Brave tab. Attempts to initialize
tab control failed before obtaining browser state, including the error
`windows sandbox failed: helper_unknown_error: setup refresh had errors`.
No new Colab cell was uploaded or executed. The release notebook contains a
bounded GPU trial using the pinned Qwen2.5-3B checkpoint, two recursive levels
and a 40-call ceiling; its small plan normally needs 30 calls.

The earlier v0.7 results and archives are unchanged. They do not validate this
new implementation's real-model behaviour.
