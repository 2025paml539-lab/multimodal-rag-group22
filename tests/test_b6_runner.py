"""B6 test: the experiment runner runs all eight experiments offline (hash embedder, fake LLM) and writes its files."""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestRunner(unittest.TestCase):
    def test_all_experiments_write_their_files(self):
        from rag import Element, Modality
        from rag.index import _element_to_dict
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "cache" / "parsed").mkdir(parents=True)
            els = [Element("loan_policy", 1, Modality.TEXT, "A minimum credit score of 700 is required for personal loans.", metadata={"el": 0}),
                   Element("loan_policy", 1, Modality.TABLE, "Loan type | Max amount\nHome loan | 50000000", metadata={"el": 1}),
                   Element("fraud_report", 2, Modality.IMAGE, "Fraud cases per quarter 900 1100 1250 980", image_path="x.png", metadata={"el": 0}),
                   Element("fraud_report", 1, Modality.TEXT, "UPI lost the most money to fraud, 310 lakh INR.", metadata={"el": 0})]
            (d / "cache" / "parsed" / "all.json").write_text(json.dumps([_element_to_dict(e) for e in els]), encoding="utf-8")
            (d / "golden").mkdir()
            rows = [("G-1", "minimum credit score for personal loans", "text", "700", "loan_policy", 1, "dev"),
                    ("G-2", "maximum amount for a home loan", "table", "50000000", "loan_policy", 1, "test"),
                    ("G-3", "fraud cases third quarter", "image", "1250", "fraud_report", 2, "dev"),
                    ("G-4", "which channel lost the most money to fraud", "text", "UPI", "fraud_report", 1, "test")]
            (d / "golden" / "golden_t.jsonl").write_text("\n".join(json.dumps(dict(zip(
                ("id", "question", "modality", "answer", "doc_id", "page", "split"), r))) for r in rows), encoding="utf-8")
            env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
            proc = subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "scripts" / "run_experiments.py"),
                                   "--index-path", str(d / "none"), "--cache", str(d / "cache"), "--results", str(d / "res"),
                                   "--golden", str(d / "golden"), "--embedder", "hash", "--llm", "fake"],
                                  capture_output=True, text=True, env=env, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stderr[-800:])
            res = d / "res"
            for name in ("e1_inventory.json", "e2_chunking.csv", "e3_retrieval.csv", "e4_transcript.md",
                         "e5_answers.csv", "e6_benchmark.json", "e7_ablation.csv", "e8_iterations.md"):
                self.assertTrue((res / name).exists(), name)
            e6 = json.loads((res / "e6_benchmark.json").read_text())
            self.assertEqual(e6["consistency_identical_answers"], 1.0)
            self.assertEqual(e6["runs_per_query"], 5)
            e5 = list(csv.DictReader(open(res / "e5_answers.csv")))
            self.assertEqual(len(e5), 4)
            self.assertEqual(len(list(csv.DictReader(open(res / "e7_ablation.csv")))), 5)

    def test_unknown_experiment_is_rejected(self):
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_experiments.py"), "--experiments", "E99"],
                              capture_output=True, text=True, env=env, timeout=60)
        self.assertNotEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
