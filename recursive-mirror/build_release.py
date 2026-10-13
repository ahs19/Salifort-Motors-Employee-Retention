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


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_payloads(root: Path = ROOT) -> dict[str, bytes]:
    selected = [root / "pyproject.toml", root / "README.md", root / "build_release.py"]
    for folder, pattern in (("recursive_mirror", "*.py"), ("tests", "test_*.py"),
                            ("examples", "*.json")):
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
    manifest = {"format": "recursive-mirror-release-v1", "version": "0.8.0",
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

    markdown("""# Digital Mirror v0.8: recursive core\n
This notebook contains the complete reviewed source archive. It creates a fresh
folder, checks the archive, runs software tests, then offers a **bounded GPU LLM
trial**. The scripted demo is not model evidence. Select a free GPU runtime for
the model trial; no paid upgrade is required or performed by these cells.

Run cells individually in order. Keep the output and export partial evidence
even if the real-model trial fails. This small arithmetic canary cannot establish
general self-improvement or subjective awareness.
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
        SESSION = Path("/content") / ("recursive_mirror_v080_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
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
        import subprocess
        def captured(label, command, timeout=120):
            print("Starting:", label, flush=True)
            log = SESSION / (label + ".log")
            # Direct file descriptors preserve partial output even if the cell
            # is interrupted before the child exits. Never overwrite a log.
            handle = log.open("x", encoding="utf-8")
            try:
                completed = subprocess.run(command, cwd=PROJECT, stdout=handle,
                                           stderr=subprocess.STDOUT, text=True, timeout=timeout)
            finally:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
                print(log.read_text(encoding="utf-8", errors="replace"))
            if completed.returncode:
                raise RuntimeError(f"{label} exited {completed.returncode}; log preserved in {SESSION}")
            return completed
        captured("unit-tests", [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"])
        captured("scripted-demo", [sys.executable, "-B", "-m", "recursive_mirror", "demo", "--output", str(SESSION / "scripted_demo")])
        print("Software checks complete. The scripted demo is NOT an LLM experiment.")
    ''')
    markdown("""## Prepare the real-model GPU trial\n
This cell checks for a GPU before installing inference dependencies. If no GPU
is available, stop here. Keep the unit-test results; do not silently run the model
on a CPU or purchase compute. The next cell may download the pinned Qwen 3B
checkpoint into Colab's cache and makes at most 40 calls (the normal plan uses 30).
""")
    code('''
        captured("gpu-status", ["nvidia-smi"], timeout=30)
        import torch
        assert torch.cuda.is_available(), "A GPU is required for this Colab model trial"
        print({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "python": sys.version})
        captured("inference-dependencies", [sys.executable, "-m", "pip", "install", "-q",
                 "transformers==4.57.6", "accelerate>=0.33,<2", "bitsandbytes>=0.45,<1"], timeout=600)
        from importlib.metadata import version
        runtime_info = {"python": sys.version, "gpu": torch.cuda.get_device_name(0),
                        "compute_capability": list(torch.cuda.get_device_capability(0)),
                        "versions": {name: version(name) for name in ["torch", "transformers", "accelerate", "bitsandbytes"]}}
        (SESSION / "runtime-info.json").write_text(json.dumps(runtime_info, indent=2), encoding="utf-8")
        print(json.dumps(runtime_info, indent=2))
    ''')
    code('''
        os.environ.setdefault("HF_HOME", "/content/hf_cache")
        REAL_OUTPUT = SESSION / "real_trial"
        MODEL = "Qwen/Qwen2.5-3B-Instruct"
        REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
        captured("real-model-trial", [sys.executable, "-B", "-m", "recursive_mirror", "run",
                 "--plan", str(PROJECT / "examples" / "canary.json"), "--output", str(REAL_OUTPUT),
                 "--model", MODEL, "--revision", REVISION, "--allow-download", "--device", "cuda", "--four-bit",
                 "--max-depth", "2", "--max-calls", "40"], timeout=1800)
        result = json.loads((REAL_OUTPUT / "result.json").read_text())
        print(json.dumps({key: result[key] for key in ["status", "calls", "transfer_improved",
                         "initial_final_metrics", "candidate_final_metrics", "claim"]}, indent=2))
    ''')
    markdown("""## Export evidence, including a failed or interrupted trial\n
Run this once while the VM still has its files. It packages this new session's
source, logs and results. Downloading the ZIP preserves evidence beyond Colab's
temporary runtime. It does not turn a failed test into a successful one.
""")
    code('''
        from google.colab import files
        export = Path("/content") / (SESSION.name + "_evidence.zip")
        with zipfile.ZipFile(export, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(SESSION.rglob("*")):
                if path.is_symlink():
                    raise RuntimeError("Unexpected symlink in evidence session")
                if path.is_file():
                    bundle.write(path, path.relative_to(SESSION).as_posix())
        digest = hashlib.sha256(export.read_bytes()).hexdigest()
        print(json.dumps({"event": "evidence_export", "path": str(export), "bytes": export.stat().st_size, "sha256": digest}))
        files.download(str(export))
    ''')
    return {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}, "colab": {"name": "Digital_Mirror_Recursive_v0_8.ipynb"}}}


def build(output: Path) -> dict:
    output = output.resolve()
    output.relative_to(ROOT)
    payloads = source_payloads()
    archive = archive_bytes(payloads)
    notebook = make_notebook(archive)
    output.mkdir(parents=True, exist_ok=False)
    archive_path = output / "digital-mirror-recursive-v0.8.0.zip"
    archive_path.write_bytes(archive)
    notebook_path = output / "Digital_Mirror_Recursive_v0_8.ipynb"
    notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    receipt = {"source_zip": str(archive_path), "source_sha256": sha(archive),
               "source_bytes": len(archive), "notebook": str(notebook_path),
               "notebook_sha256": sha(notebook_path.read_bytes()), "packaged_files": len(payloads)}
    (output / "BUILD_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))
