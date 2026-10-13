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

    def test_trial_remains_gpu_only_bounded_and_revision_pinned(self):
        notebook = make_notebook(archive_bytes(source_payloads()))
        sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        setup = next(s for s in sources if 'captured("gpu-status"' in s)
        trial = next(s for s in sources if 'captured("real-model-trial"' in s)
        self.assertLess(setup.index('captured("gpu-status"'), setup.index('"pip", "install"'))
        self.assertIn('"--device", "cuda"', trial)
        self.assertIn('"--max-depth", "2", "--max-calls", "40"', trial)
        self.assertIn('"--revision", REVISION', trial)
        self.assertIn('aa8e72537993ba99e69dfaafa59ed015b17504d1', trial)
        self.assertIn('timeout=1800', trial)

    def test_interrupted_cell_preserves_child_log_and_refuses_overwrite(self):
        notebook = make_notebook(archive_bytes(source_payloads()))
        source = next("".join(c["source"]) for c in notebook["cells"]
                      if c["cell_type"] == "code" and "def captured(" in "".join(c["source"]))
        tree = ast.parse(source)
        definition = ast.Module(body=[node for node in tree.body
                                     if isinstance(node, (ast.Import, ast.FunctionDef))], type_ignores=[])
        with tempfile.TemporaryDirectory() as directory:
            context = {"SESSION": Path(directory), "PROJECT": Path(directory), "os": os}
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
