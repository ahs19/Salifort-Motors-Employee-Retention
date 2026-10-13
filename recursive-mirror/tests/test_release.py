"""Check transport/build artefacts without Colab, packages or model execution."""
from contextlib import redirect_stdout
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from build_release import archive_bytes, make_notebook, source_payloads


class ReleaseTests(unittest.TestCase):
    def test_archive_inventory_and_hashes_match_sources(self):
        payloads = source_payloads()
        blob = archive_bytes(payloads)
        self.assertEqual(blob, archive_bytes(payloads))
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            self.assertIsNone(archive.testzip())
            manifest = json.loads(archive.read("MANIFEST.json"))
            self.assertEqual(set(archive.namelist()), set(manifest["files"]) | {"MANIFEST.json"})
            for name, metadata in manifest["files"].items():
                data = archive.read(name)
                self.assertEqual(len(data), metadata["bytes"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), metadata["sha256"])
                self.assertEqual(data, payloads[name])

    def test_notebook_bootstrap_verifies_and_extracts_exact_sources(self):
        payloads = source_payloads()
        notebook = make_notebook(archive_bytes(payloads))
        cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len({c["id"] for c in notebook["cells"]}), len(notebook["cells"]))
        for cell in cells:
            compile("".join(cell["source"]), "generated-cell", "exec")
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            bootstrap = "".join(cells[0]["source"])
            bootstrap = bootstrap.replace('Path("/content")', f"Path({directory!r})")
            namespace = {}
            try:
                with redirect_stdout(io.StringIO()):
                    exec(compile(bootstrap, "bootstrap-local-test", "exec"), namespace)
                project = namespace["PROJECT"]
                for name, data in payloads.items():
                    self.assertEqual((project / name).read_bytes(), data)
                self.assertEqual(namespace["SOURCE_ZIP_SHA256"], hashlib.sha256(archive_bytes(payloads)).hexdigest())
            finally:
                os.chdir(previous)

    def test_notebook_runs_a_bounded_pinned_gpu_experiment_in_safe_order(self):
        notebook = make_notebook(archive_bytes(source_payloads()))
        sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]

        def at(marker):
            return next(i for i, s in enumerate(sources) if marker in s)

        settings = sources[at('REVISION = "')]
        self.assertIn('MODEL = "Qwen/Qwen2.5-3B-Instruct"', settings)
        self.assertIn('REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"', settings)
        gpu = sources[at('captured("gpu-status"')]
        self.assertLess(gpu.index('captured("gpu-status"'), gpu.index('"pip", "install"'))
        self.assertLess(gpu.index('"pip", "install"'), gpu.index('"gpu-integration-test"'))
        self.assertIn('RECURSIVE_MIRROR_TEST_DEVICE="cuda"', gpu)
        self.assertIn('"--revision", REVISION, "--allow-download", "--device", "cuda"', gpu)
        self.assertIn("CAPABILITY[0] >= 8", gpu)
        # Software tests, then GPU check, then preflight, then the experiment, then export.
        order = [at('"unit-tests"'), at('captured("gpu-status"'), at('"preflight-run"'),
                 at('"experiment", "--output"'), at("EXPORT = export_evidence()")]
        self.assertEqual(order, sorted(order))
        self.assertEqual(order[-1], len(sources) - 1)
        experiment = sources[at('"experiment", "--output"')]
        for flag in ('"--max-calls", str(MAX_CALLS_PER_RUN)', '"--max-seconds", str(PER_RUN_SECONDS)', "*MODEL_ARGS"):
            self.assertIn(flag, experiment)
        self.assertNotIn("raise", experiment)  # a failed run must still reach the export cell

    def test_notebook_budget_covers_its_experiment_and_the_shipped_plan(self):
        from recursive_mirror.core import Config, RecursiveMirror
        from recursive_mirror.runtime import load_plan
        from recursive_mirror.tasks import generate_plan
        model = type("M", (), {"identity": "budget-check"})()
        plan = generate_plan(2026, rounds=3, discovery=12, validation=24, final=48)
        engine = RecursiveMirror(model, plan, Config(max_depth=3, max_calls=720))
        self.assertEqual(sum(engine._round_cost(r) for r in plan.rounds) + engine._final_reserve, 711)
        shipped = load_plan(Path(__file__).resolve().parents[1] / "examples" / "headroom.json")
        engine = RecursiveMirror(model, shipped, Config(max_depth=2, max_calls=240))
        self.assertEqual(sum(engine._round_cost(r) for r in shipped.rounds[:2]) + engine._final_reserve, 218)

    def test_committed_release_embeds_exactly_the_current_sources(self):
        releases = sorted((Path(__file__).resolve().parents[1] / "release").glob("*/BUILD_RECEIPT.json"))
        if not releases:
            self.skipTest("no committed release in this checkout (e.g. inside the packaged archive)")
        self.assertEqual(len(releases), 1, "keep only the current release under release/")
        receipt = json.loads(releases[0].read_text(encoding="utf-8"))
        blob = archive_bytes(source_payloads())
        self.assertEqual(receipt["source_sha256"], hashlib.sha256(blob).hexdigest(),
                         "sources changed since the release was built: rebuild it with build_release.py")
        notebook = releases[0].parent / receipt["notebook"]
        self.assertEqual(hashlib.sha256(notebook.read_bytes()).hexdigest(), receipt["notebook_sha256"])
        self.assertEqual((releases[0].parent / receipt["source_zip"]).read_bytes(), blob)
        self.assertEqual(json.loads(notebook.read_text(encoding="utf-8")), make_notebook(blob))

    def test_interrupted_cell_preserves_child_log_and_refuses_overwrite(self):
        notebook = make_notebook(archive_bytes(source_payloads()))
        source = next("".join(c["source"]) for c in notebook["cells"]
                      if c["cell_type"] == "code" and "def captured(" in "".join(c["source"]))
        tree = ast.parse(source)
        definition = ast.Module(body=[node for node in tree.body
                                     if isinstance(node, (ast.Import, ast.FunctionDef))], type_ignores=[])
        with tempfile.TemporaryDirectory() as directory:
            context = {"SESSION": Path(directory), "PROJECT": Path(directory), "os": os, "ENV": dict(os.environ)}
            exec(compile(definition, "capture-helper-test", "exec"), context)

            def interrupted(*args, **kwargs):
                self.assertNotIn("capture_output", kwargs)
                kwargs["stdout"].write("partial child stderr retained\n")
                kwargs["stdout"].flush()
                raise KeyboardInterrupt()

            with patch.object(context["subprocess"], "run", side_effect=interrupted), redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    context["captured"]("interrupted", ["not-executed"])
                with self.assertRaises(FileExistsError):
                    context["captured"]("interrupted", ["not-executed"])
            self.assertEqual((Path(directory) / "interrupted.log").read_text(), "partial child stderr retained\n")


if __name__ == "__main__":
    unittest.main()
