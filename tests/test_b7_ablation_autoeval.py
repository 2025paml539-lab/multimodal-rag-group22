"""Tests: the five ablation prompt styles from the brief, and the automatic evaluation (LLM-as-judge)."""
import unittest

from rag import Element, FakeLLMClient, Hit, Modality
from rag.auto_eval import cohens_kappa, compare_with_golden, judge
from rag.prompts import ABLATION_STYLES, REFUSAL, render_prompt


def hits():
    return [Hit(Element("loan_policy", 1, Modality.TABLE, "Loan type | Rate\nHome loan | 8.6"), 1.0, 1)]


class TestAblationStyles(unittest.TestCase):
    def test_the_five_styles_named_in_the_brief(self):
        self.assertEqual(ABLATION_STYLES, ("zero_shot", "fstring", "three_shot", "jinja2", "create"))

    def test_every_style_renders_question_context_citation_and_refusal(self):
        for style in ABLATION_STYLES:
            text = render_prompt(style, "What is the home loan rate?", hits(), {"doc_types": ["master_direction"]})
            self.assertIn("What is the home loan rate?", text, style)
            self.assertIn("Home loan | 8.6", text, style)
            self.assertIn("[loan_policy, p.1]", text, style)
            self.assertIn(REFUSAL, text, style)

    def test_three_shot_has_three_examples_and_zero_shot_none(self):
        self.assertEqual(render_prompt("three_shot", "q", hits()).count("Example "), 3)
        self.assertNotIn("Example", render_prompt("zero_shot", "q", hits()))

    def test_fstring_style_does_not_need_jinja(self):
        text = render_prompt("fstring", "q", hits())
        self.assertTrue(text.startswith("Answer the question using only the context"))


class TestJudge(unittest.TestCase):
    def test_valid_and_invalid_judge_replies(self):
        ok = FakeLLMClient(['{"supported": true, "answers_question": true}'])
        self.assertEqual(judge(ok, "q", "a", hits()), {"supported": True, "answers_question": True})
        wrapped = FakeLLMClient(['Sure! {"supported": false, "answers_question": true} done'])
        self.assertEqual(judge(wrapped, "q", "a", hits())["supported"], False)
        self.assertIsNone(judge(FakeLLMClient(["I cannot say"]), "q", "a", hits()))

    def test_judge_prompt_does_not_contain_a_golden_answer_field(self):
        llm = FakeLLMClient(['{"supported": true, "answers_question": true}'])
        judge(llm, "What is the rate?", "8.6 [loan_policy, p.1]", hits())
        prompt = llm.calls[0]["prompt"]
        self.assertIn("What is the rate?", prompt)
        self.assertNotIn("golden", prompt.lower())


class TestAgreement(unittest.TestCase):
    def test_kappa_values(self):
        self.assertEqual(cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]), 1.0)
        self.assertEqual(cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0]), 0.0)
        self.assertIsNone(cohens_kappa([1, 1], [1, 1]))      # no variation: undefined, reported as None
        self.assertIsNone(cohens_kappa([], []))

    def test_compare_counts_disagreements_and_invalid_replies(self):
        rows = [{"id": "a", "correct": 1, "judge_ok": 1}, {"id": "b", "correct": 0, "judge_ok": 1},
                {"id": "c", "correct": 1, "judge_ok": 0}, {"id": "d", "correct": 0, "judge_ok": 0},
                {"id": "e", "correct": 1, "judge_ok": None}]
        out = compare_with_golden(rows)
        self.assertEqual((out["questions"], out["compared"], out["judge_invalid_replies"]), (5, 4, 1))
        self.assertEqual(out["agreement"], 0.5)
        self.assertEqual(out["judge_says_ok_golden_says_wrong"], ["b"])
        self.assertEqual(out["judge_says_wrong_golden_says_ok"], ["c"])


if __name__ == "__main__":
    unittest.main()
