"""B4 tests: prompt variants and evaluation metrics (no GPU, no network)."""
import json
import tempfile
import unittest
from pathlib import Path

from rag import Element, FakeLLMClient, Hit, Modality
from rag.eval_metrics import (cites_retrieved, correctness, faithfulness, is_refusal, load_golden, mrr,
                              normalise, parse_citations, recall_at_k, summarise)
from rag.prompts import REFUSAL, VARIANTS, load_ontology, render_prompt


def hits():
    e1 = Element("loan_policy", 1, Modality.TABLE, "Loan type | Rate\nHome loan | 8.6", metadata={"el": 0})
    e2 = Element("fraud_report", 2, Modality.IMAGE, "Fraud cases per quarter 900 1100 1250 980", image_path="x.png")
    return [Hit(e1, 0.9, 1), Hit(e2, 0.5, 2)]


class TestPrompts(unittest.TestCase):
    def test_every_variant_renders_with_citation_and_refusal_instruction(self):
        for v in VARIANTS:
            text = render_prompt(v, "What is the home loan rate?", hits(), load_ontology())
            self.assertIn("What is the home loan rate?", text, v)
            self.assertIn("[loan_policy, p.1]", text, v)
            self.assertIn("Home loan | 8.6", text, v)
            self.assertIn(REFUSAL, text, v)
            self.assertTrue("p.N" in text or "page" in text, v)

    def test_unknown_variant_is_rejected(self):
        with self.assertRaises(ValueError):
            render_prompt("v9_missing", "q", hits())

    def test_works_without_ontology(self):
        self.assertIn("q?", render_prompt("v2_citation_rules", "q?", hits(), None))


class TestRetrievalMetrics(unittest.TestCase):
    def test_recall_and_mrr(self):
        h = hits()
        self.assertEqual(recall_at_k(h, "loan_policy", 1, k=1), 1.0)
        self.assertEqual(recall_at_k(h, "fraud_report", 2, k=1), 0.0)
        self.assertEqual(recall_at_k(h, "fraud_report", 2, k=2), 1.0)
        self.assertEqual(mrr(h, "fraud_report", 2), 0.5)
        self.assertEqual(mrr(h, "other", 9), 0.0)


class TestAnswerMetrics(unittest.TestCase):
    def test_correctness_normalises_numbers_and_citations(self):
        self.assertEqual(correctness("The Q3 value is 1,250 [fraud_report, p.2].", "1250"), 1.0)
        self.assertEqual(correctness("It is 980 [fraud_report, p.2].", "1250"), 0.0)
        self.assertEqual(correctness("The rate is 8.6 % p.a.", "8.6"), 1.0)
        self.assertEqual(correctness("The rate is 18.6", "8.6"), 0.0)

    def test_equivalent_wording_is_not_marked_wrong(self):
        # the four answers from lab session S3 that were correct but scored wrong
        self.assertEqual(correctness("within three working days [D021, p.30].", "3 working days"), 1.0)
        self.assertEqual(correctness("The maximum liability is \u20b95,000 [D083, p.15].", "Rs 5,000"), 1.0)
        self.assertEqual(correctness("limit is Rs. 25,000", "Rs 25,000"), 1.0)
        self.assertEqual(correctness("it is INR 25000", "Rs 25,000"), 1.0)
        self.assertEqual(correctness("the category is Home loan [D149, p.4].", "Education loan"), 0.0)
        self.assertEqual(correctness("within thirteen working days", "3 working days"), 0.0)

    def test_refusing_a_gold_answer_is_wrong_and_judge_is_used_otherwise(self):
        self.assertEqual(correctness("Not found in the documents.", "1250"), 0.0)
        yes = FakeLLMClient(["yes"])
        self.assertEqual(correctness("About twelve fifty", "1250", judge=yes), 1.0)
        no = FakeLLMClient(["no"])
        self.assertEqual(correctness("About eleven", "1250", judge=no), 0.0)

    def test_citations(self):
        self.assertEqual(parse_citations("A [loan_policy, p.1] B [fraud_report, p. 2]"), [("loan_policy", 1), ("fraud_report", 2)])
        self.assertTrue(cites_retrieved("x [loan_policy, p.1]", hits()))
        self.assertFalse(cites_retrieved("x [loan_policy, p.9]", hits()))
        self.assertFalse(cites_retrieved("no citation", hits()))

    def test_faithfulness(self):
        h = hits()
        self.assertEqual(faithfulness("Home loan rate is 8.6 [loan_policy, p.1].", h), 1.0)
        self.assertEqual(faithfulness("Not found in the documents.", h), 1.0)
        self.assertEqual(faithfulness("Mortgage rate equals 12.9 according to regulator.", h), 0.0)
        self.assertEqual(faithfulness("", h), 0.0)
        self.assertTrue(is_refusal("Not found in the documents."))
        self.assertEqual(normalise("1,250."), "1250")

    def test_summarise(self):
        rows = [{"modality": "text", "split": "dev", "recall": 1.0, "mrr": 1.0, "correct": 1.0, "faithful": 1.0},
                {"modality": "image", "split": "test", "recall": 0.0, "mrr": 0.0, "correct": 0.0, "faithful": 1.0}]
        s = summarise(rows)
        self.assertEqual(s["all"]["recall"], 0.5)
        self.assertEqual(s["modality:image"]["correct"], 0.0)


class TestGoldenLoader(unittest.TestCase):
    row = {"id": "A-1", "question": "q", "modality": "text", "answer": "a", "doc_id": "D1", "page": 1, "split": "dev"}

    def _dir(self, *files):
        d = tempfile.mkdtemp()
        for i, rows in enumerate(files):
            Path(d, f"golden_{i}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return d

    def test_merges_files(self):
        rows = load_golden(self._dir([self.row], [{**self.row, "id": "B-1"}]))
        self.assertEqual([r["id"] for r in rows], ["A-1", "B-1"])

    def test_duplicate_id_fails(self):
        with self.assertRaises(ValueError):
            load_golden(self._dir([self.row], [self.row]))

    def test_missing_split_and_bad_split_fail(self):
        bad = dict(self.row); del bad["split"]
        with self.assertRaises(ValueError):
            load_golden(self._dir([bad]))
        with self.assertRaises(ValueError):
            load_golden(self._dir([{**self.row, "split": "val"}]))

    def test_repo_golden_file_is_valid(self):
        rows = load_golden(str(Path(__file__).resolve().parent.parent / "eval"))
        self.assertGreaterEqual(len(rows), 8)


if __name__ == "__main__":
    unittest.main()
