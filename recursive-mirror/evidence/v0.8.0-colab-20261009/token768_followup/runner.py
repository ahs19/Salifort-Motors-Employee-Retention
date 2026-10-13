import json, sys
from dataclasses import asdict
from recursive_mirror.core import Config
from recursive_mirror.runtime import HuggingFaceBackend, execute, load_plan
model = HuggingFaceBackend("Qwen/Qwen2.5-3B-Instruct",
    revision="aa8e72537993ba99e69dfaafa59ed015b17504d1",
    allow_download=False, device="cuda", four_bit=True)
config = Config(max_depth=2, max_calls=40, max_new_tokens=768)
result = execute(model, load_plan(sys.argv[1]), sys.argv[2], config)
print(json.dumps({key: asdict(result)[key] for key in
    ["status", "calls", "transfer_improved", "initial_final_metrics",
     "candidate_final_metrics", "claim"]}, indent=2), flush=True)
