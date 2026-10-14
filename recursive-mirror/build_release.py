"""Build a bounded source ZIP and a self-contained Colab notebook; no model calls."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import stat
import textwrap
import zipfile


ROOT = Path(__file__).resolve().parent
VERSION = "0.11.0"
NOTEBOOK_NAME = "Digital_Mirror_Recursive_v0_11.ipynb"
ARCHIVE_NAME = f"digital-mirror-recursive-v{VERSION}.zip"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_payloads(root: Path = ROOT) -> dict[str, bytes]:
    selected = [root / "pyproject.toml", root / "README.md", root / "build_release.py"]
    for folder, pattern in (("recursive_mirror", "*.py"), ("tests", "test_*.py"),
                            ("examples", "*.json"), ("docs", "*.md")):
        selected.extend(sorted((root / folder).glob(pattern)))
    if (root / "VALIDATION.md").is_file():
        selected.append(root / "VALIDATION.md")
    payloads = {}
    for path in selected:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Not a regular release source: {path}")
        name = path.relative_to(root).as_posix()
        data = path.read_bytes()
        if name.endswith(".py"):
            compile(data, name, "exec")
        payloads[name] = data
    manifest = {"format": "recursive-mirror-release-v1", "version": VERSION,
                "files": {name: {"bytes": len(data), "sha256": sha(data)}
                          for name, data in sorted(payloads.items())}}
    payloads["MANIFEST.json"] = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    return payloads


def archive_bytes(payloads: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payloads.items()):
            item = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(item, data)
    return buffer.getvalue()


def make_notebook(archive: bytes) -> dict:
    cells = []

    def markdown(text: str) -> None:
        cells.append({"cell_type": "markdown", "id": f"cell-{len(cells)}",
                      "metadata": {}, "source": text.splitlines(True)})

    def code(text: str) -> None:
        text = textwrap.dedent(text).lstrip("\n")
        compile(text, "colab-cell", "exec")
        cells.append({"cell_type": "code", "id": f"cell-{len(cells)}", "metadata": {}, "source": text.splitlines(True),
                      "outputs": [], "execution_count": None})

    markdown("""# Digital Mirror v0.11: does the AI improve itself?\n
**How to run.** *Runtime → Change runtime type →* pick a **GPU** (T4 works; L4 or A100 is faster) → *Save*.
Then *Runtime → **Run all***. Keep this tab open. The last cell downloads `…_evidence.zip`:
upload that file to Claude.

What the cells do, in order:
1. verify the embedded, hash-checked source archive and run the software tests;
2. detect the GPU, install the pinned inference libraries, and test the real
   inference stack on the GPU with a tiny local model;
3. run a short preflight with the real model (Qwen2.5-3B-Instruct, pinned
   revision) and print protocol diagnostics;
4. run the experiment: for each of several independent task plans, the agent
   recursively examines itself, adds one rule at a time to its strategy for its
   weakest task types and recalibrates its self-model, keeping only changes that
   pass paired tests on fresh tasks; then a held-out final test. A control arm
   repeats this from a hand-written step-by-step prompt;
5. pool the held-out results across plans and export all evidence.

Expected time with the default settings: about 40 minutes on an A100, several hours on a T4
(use fewer `SEEDS` or `ROUNDS` there).
Use the settings cell to shorten it (for example fewer `SEEDS`).
If any cell stops with an error, run the **last cell** by hand: it exports
everything produced so far, including logs of the failure.

A positive result shows measured, transferable improvement of this model's
inference-time strategy or self-model on these task types. It does not show
general self-improvement, changed weights, or subjective awareness.
""")
    encoded = base64.b64encode(archive).decode("ascii")
    bootstrap = '''
        import base64, hashlib, io, json, os, stat, sys, uuid, zipfile
        from pathlib import Path, PurePosixPath
        from datetime import datetime, timezone
        SOURCE_ZIP_SHA256 = "__SOURCE_SHA__"
        blob = base64.b64decode("__SOURCE_BASE64__", validate=True)
        assert hashlib.sha256(blob).hexdigest() == SOURCE_ZIP_SHA256, "Source archive mismatch"
        z = zipfile.ZipFile(io.BytesIO(blob))
        infos = z.infolist()
        names = [i.filename for i in infos]
        assert len(names) == len(set(names)) and len(names) < 100, "Unexpected archive inventory"
        assert sum(i.file_size for i in infos) < 4_000_000, "Archive exceeds source size bound"
        for item in infos:
            path = PurePosixPath(item.filename)
            assert not path.is_absolute() and ".." not in path.parts and ":" not in item.filename
            assert "\\\\" not in item.filename and not item.is_dir()
            assert stat.S_IFMT(item.external_attr >> 16) == stat.S_IFREG, "Nonregular archive member"
        data = {name: z.read(name) for name in names}
        manifest = json.loads(data["MANIFEST.json"])
        assert manifest["format"] == "recursive-mirror-release-v1"
        assert set(data) == set(manifest["files"]) | {"MANIFEST.json"}
        for name, meta in manifest["files"].items():
            assert len(data[name]) == meta["bytes"]
            assert hashlib.sha256(data[name]).hexdigest() == meta["sha256"], name
        SESSION = Path("/content") / ("recursive_mirror_v0110_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
        SESSION.mkdir(exist_ok=False)
        PROJECT = SESSION / "project"
        PROJECT.mkdir()
        for name, content in data.items():
            target = PROJECT.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(content)
        with (SESSION / "source.zip").open("xb") as handle:
            handle.write(blob)
        os.chdir(PROJECT)
        print(json.dumps({"event": "source_verified", "session": str(SESSION), "source_sha256": SOURCE_ZIP_SHA256, "files": len(data)}))
    '''
    code(bootstrap.replace("__SOURCE_SHA__", sha(archive)).replace("__SOURCE_BASE64__", encoded))
    code('''
        # Experiment settings: edit before "Run all" for a shorter or longer run.
        MODEL = "Qwen/Qwen2.5-3B-Instruct"
        REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"  # pinned commit of MODEL; change both together
        SEEDS = "2026,2027,2028"      # independent task plans; fewer seeds = shorter run, weaker evidence
        RUN_CONTROL = True            # also start from a hand-written step-by-step prompt
        ROUNDS, DISCOVERY, VALIDATION, FINAL = 4, 12, 24, 48   # rules accumulate over levels
        MAX_CALLS_PER_RUN = 1100      # 4 levels x 3 candidates need at most 1080 calls with these sizes
        TOTAL_HOURS = 4.0             # wall-clock budget shared by all runs of the main experiment
        FORCE_AFTER_WEAK_PREFLIGHT = False  # True: run the experiment even if the preflight looks uninformative
        print({"model": MODEL, "seeds": SEEDS, "control": RUN_CONTROL, "hours": TOTAL_HOURS})
    ''')
    code('''
        import subprocess, time, shutil
        sys.path.insert(0, str(PROJECT))
        ENV = dict(os.environ, HF_HOME="/content/hf_cache", USE_TF="0", TOKENIZERS_PARALLELISM="false")

        def captured(label, command, timeout=120, env=None):
            print("Starting:", label, flush=True)
            log = SESSION / (label + ".log")
            # Direct file descriptors preserve partial output even if the cell
            # is interrupted before the child exits. Never overwrite a log.
            handle = log.open("x", encoding="utf-8")
            try:
                completed = subprocess.run(command, cwd=PROJECT, stdout=handle, env=env or ENV,
                                           stderr=subprocess.STDOUT, text=True, timeout=timeout)
            finally:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
                print(log.read_text(encoding="utf-8", errors="replace"))
            if completed.returncode:
                raise RuntimeError(f"{label} exited {completed.returncode}; log preserved in {SESSION}")
            return completed

        def streamed(label, command, timeout):
            """Run a long child process and show its log live; returns the exit code."""
            print("Starting:", label, flush=True)
            log = SESSION / (label + ".log")
            handle = log.open("x", encoding="utf-8")
            process = subprocess.Popen(command, cwd=PROJECT, stdout=handle, stderr=subprocess.STDOUT,
                                       text=True, env=ENV)
            shown, deadline = 0, time.monotonic() + timeout
            try:
                while process.poll() is None:
                    time.sleep(10)
                    text = log.read_text(encoding="utf-8", errors="replace")
                    print(text[shown:], end="", flush=True)
                    shown = len(text)
                    if time.monotonic() > deadline:
                        print(f"\\n{label}: notebook timeout reached; stopping the child", flush=True)
                        process.terminate()
                        process.wait(120)
            except KeyboardInterrupt:
                process.terminate()
                process.wait(120)
                raise
            finally:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
                text = log.read_text(encoding="utf-8", errors="replace")
                print(text[shown:], flush=True)
            return process.returncode

        def export_evidence():
            """Zip the whole session (sources, logs, journals, results) and offer it for download."""
            stamp = datetime.now(timezone.utc).strftime("%H%M%S")
            export = Path("/content") / f"{SESSION.name}_evidence_{stamp}.zip"
            with zipfile.ZipFile(export, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
                for path in sorted(SESSION.rglob("*")):
                    if path.is_symlink():
                        raise RuntimeError("Unexpected symlink in evidence session")
                    if path.is_file():
                        bundle.write(path, path.relative_to(SESSION).as_posix())
            digest = hashlib.sha256(export.read_bytes()).hexdigest()
            print(json.dumps({"event": "evidence_export", "path": str(export), "bytes": export.stat().st_size,
                              "sha256": digest}))
            try:
                from google.colab import files
                files.download(str(export))
            except ImportError:
                print("Not running in Colab: download", export, "manually.")
            return export
    ''')
    code('''
        captured("unit-tests", [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"])
        captured("scripted-demo", [sys.executable, "-B", "-m", "recursive_mirror", "demo", "--output",
                                   str(SESSION / "scripted_demo")])
        print("Software checks complete. The scripted demo is NOT an LLM experiment.")
    ''')
    markdown("""## GPU, pinned libraries and a real-stack test\n
Checks for a GPU before installing anything. The inference libraries are pinned
(`transformers==4.57.6`). A tiny locally built Qwen2-architecture model then
exercises the real backend on the GPU (generation, batching, EOS, truncation)
in about a minute, before any long run.
""")
    code('''
        captured("gpu-status", ["nvidia-smi"], timeout=60)
        import torch
        assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > GPU, then Run all again."
        GPU = torch.cuda.get_device_name(0)
        CAPABILITY = torch.cuda.get_device_capability(0)
        DTYPE = "bfloat16" if CAPABILITY[0] >= 8 else "float16"   # T4 (7.5) has native FP16 only
        MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
        BATCH = 16 if MEMORY_GB >= 30 else 8
        captured("inference-dependencies", [sys.executable, "-m", "pip", "install", "-q",
                 "transformers==4.57.6", "accelerate>=0.33,<2"], timeout=900)
        from importlib.metadata import version
        runtime_info = {"python": sys.version, "gpu": GPU, "compute_capability": list(CAPABILITY),
                        "memory_gb": round(MEMORY_GB, 1), "dtype": DTYPE, "batch_size": BATCH,
                        "versions": {name: version(name) for name in ["torch", "transformers", "accelerate"]}}
        (SESSION / "runtime-info.json").write_text(json.dumps(runtime_info, indent=2), encoding="utf-8")
        print(json.dumps(runtime_info, indent=2))
        # Load the test by file path: Colab ships an unrelated top-level package named tests,
        # which shadows this project's tests folder when a test is imported by dotted name.
        captured("gpu-integration-test", [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests",
                 "-p", "test_hf_integration.py", "-v"], timeout=900, env=dict(ENV, RECURSIVE_MIRROR_TEST_DEVICE="cuda"))
        MODEL_ARGS = ["--model", MODEL, "--revision", REVISION, "--allow-download", "--device", "cuda",
                      "--dtype", DTYPE, "--batch-size", str(BATCH), "--max-input-tokens", "8192"]
    ''')
    markdown("""## Preflight with the real model (about 5 minutes)\n
Downloads the pinned model once, then runs one short level on a separate small
plan and prints diagnostics: answer-line use, invalid forecasts, truncations and
speed. It stops before the long run only if the model cannot follow the protocol
at all.
""")
    code('''
        from recursive_mirror.diagnostics import diagnose, diagnosis_markdown
        PREFLIGHT_PLAN = SESSION / "preflight-plan.json"
        PREFLIGHT = SESSION / "preflight"
        captured("preflight-plan", [sys.executable, "-B", "-m", "recursive_mirror", "make-plan", "--output",
                 str(PREFLIGHT_PLAN), "--seed", "7", "--rounds", "1", "--discovery", "6", "--validation", "3",
                 "--final", "3"])
        code = streamed("preflight-run", [sys.executable, "-B", "-m", "recursive_mirror", "run", "--plan",
                        str(PREFLIGHT_PLAN), "--output", str(PREFLIGHT), *MODEL_ARGS, "--max-depth", "1",
                        "--max-calls", "60"], timeout=3600)
        health = diagnose(PREFLIGHT)
        print(diagnosis_markdown(PREFLIGHT, health))
        answers = sum(health["answer_status"].values()) or 1
        usable = sum(health["answer_status"].get(k, 0) for k in ("answer_line", "boxed", "last_line"))
        valid = health["forecasts"].get("valid", 0) / (sum(health["forecasts"].values()) or 1)
        problems = []
        if code != 0:
            problems.append(f"the preflight run exited with code {code}")
        if usable / answers < 0.5:
            problems.append(f"only {usable}/{answers} answers were usable (the rest hit the token limit or were empty)")
        if valid < 0.5:
            problems.append(f"only {valid:.0%} of forecasts were valid percentages")
        if problems and not FORCE_AFTER_WEAK_PREFLIGHT:
            raise RuntimeError("Preflight looks uninformative: " + "; ".join(problems) + ". Run the last cell to "
                               "export the evidence and send it to Claude (or set FORCE_AFTER_WEAK_PREFLIGHT).")
        print("Preflight passed" + (" with warnings: " + "; ".join(health["warnings"] + problems)
                                    if health["warnings"] or problems else "."))
    ''')
    markdown("""## The experiment\n
Each run writes its own journal, result, checkpoint and `report.md`; a pooled
`summary.md` is rewritten after every run, so even an interrupted session keeps
usable evidence. Progress lines appear below as levels complete.
""")
    code('''
        EXPERIMENT = SESSION / "experiment"
        TOTAL_SECONDS = int(TOTAL_HOURS * 3600)  # shared out as it is spent: fast runs leave time for later ones
        command = [sys.executable, "-B", "-m", "recursive_mirror", "experiment", "--output", str(EXPERIMENT),
                   *MODEL_ARGS, "--seeds", SEEDS, "--rounds", str(ROUNDS), "--discovery", str(DISCOVERY),
                   "--validation", str(VALIDATION), "--final", str(FINAL), "--max-depth", str(ROUNDS),
                   "--max-calls", str(MAX_CALLS_PER_RUN), "--total-seconds", str(TOTAL_SECONDS)]
        if not RUN_CONTROL:
            command += ["--control-strategy", ""]
        code = streamed("experiment", command, timeout=int(TOTAL_HOURS * 3600) + 3600)
        summary = EXPERIMENT / "summary.md"
        print(summary.read_text(encoding="utf-8") if summary.exists() else "No summary was written; see experiment.log")
        if code:
            print("Some runs did not complete; their evidence is still exported by the next cell.")
    ''')
    markdown("""## Export evidence (also run this by hand after any error)\n
Packages this session's sources, logs, journals, results and summaries into one
zip and downloads it. Upload the zip to Claude for analysis.
""")
    code('''
        EXPORT = export_evidence()
    ''')
    return {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}, "accelerator": "GPU",
                         "colab": {"name": NOTEBOOK_NAME, "gpuType": "T4"}}}


def build(output: Path) -> dict:
    output = output.resolve()
    output.relative_to(ROOT)
    payloads = source_payloads()
    archive = archive_bytes(payloads)
    notebook = make_notebook(archive)
    output.mkdir(parents=True, exist_ok=False)
    archive_path = output / ARCHIVE_NAME
    archive_path.write_bytes(archive)
    notebook_path = output / NOTEBOOK_NAME
    notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    receipt = {"version": VERSION, "source_zip": archive_path.name, "source_sha256": sha(archive),
               "source_bytes": len(archive), "notebook": notebook_path.name,
               "notebook_sha256": sha(notebook_path.read_bytes()), "packaged_files": len(payloads)}
    (output / "BUILD_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))
