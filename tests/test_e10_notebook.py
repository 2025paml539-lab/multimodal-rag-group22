"""E10 script runs offline on a small saved index (hash embedder, fake LLM); the notebook file is valid JSON."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestE10(unittest.TestCase):
    def test_blocked_steps_behave_as_expected_without_an_llm(self):
        from rag import Element, HashEmbedder, HybridIndex, Modality
        with tempfile.TemporaryDirectory() as d:
            idx = HybridIndex(HashEmbedder())
            idx.add([Element("D054", 8, Modality.TEXT, "The customer must submit an OVD with current address within a period of three months.")])
            idx.save(os.path.join(d, "idx"))
            env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
            p = subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "scripts" / "e10_demo.py"), "--index-path", os.path.join(d, "idx"),
                                "--embedder", "hash", "--llm", "fake", "--results", os.path.join(d, "out")],
                               capture_output=True, text=True, env=env, timeout=120)
            self.assertEqual(p.returncode, 0, p.stderr[-600:])
            rows = json.loads(Path(d, "out", "e10_demo.json").read_text(encoding="utf-8"))
            by = {r["label"]: r for r in rows}
            self.assertEqual(len(rows), 8)
            for label in ("prompt injection", "off-topic", "personal data"):
                self.assertTrue(by[label]["as_expected"], label)
                self.assertEqual(by[label]["stage"], "input")
            self.assertTrue(Path(d, "out", "e10_demo.md").exists())


class TestNotebook(unittest.TestCase):
    def test_notebook_is_valid_and_has_no_stored_fake_outputs(self):
        nb = json.loads((ROOT / "notebooks" / "part_a_demo.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertGreaterEqual(len(code), 6)
        self.assertTrue(all(c["outputs"] == [] for c in code))   # outputs appear only after a real run in the lab


if __name__ == "__main__":
    unittest.main()
