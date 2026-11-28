"""Automatic evaluation without hand labels (Task A4 alternative approach).

LLM-as-judge with a rubric, reference-free: the judge sees only the question, the retrieved context and the answer,
never the golden answer. Its verdict is then compared with the golden-set correctness (agreement and Cohen's kappa).
"""
from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Sequence
from .elements import Hit
from .llm import LLMClient

RUBRIC = """You are grading an answer from a banking question-answering system.
Use ONLY the context. Do not use outside knowledge.
1. supported: every fact and number in the answer appears in the context.
2. answers_question: the answer addresses what the question asks. A refusal ("Not found in the documents.") does not.
Return one JSON object and nothing else: {{"supported": true or false, "answers_question": true or false}}

Context:
{context}

Question: {question}
Answer: {answer}
JSON:"""


def judge(llm: LLMClient, question: str, answer: str, hits: Sequence[Hit]) -> Optional[Dict[str, bool]]:
    """Return {"supported": bool, "answers_question": bool}, or None if the judge reply is not valid JSON."""
    context = "\n".join(f"[{h.element.doc_id}, p.{h.element.page}] {h.element.text}" for h in hits)
    raw = llm.generate(RUBRIC.format(context=context, question=question, answer=answer),
                       temperature=0.0, max_tokens=40).text
    try:
        obj = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        return {"supported": bool(obj["supported"]), "answers_question": bool(obj["answers_question"])}
    except Exception:
        return None


def cohens_kappa(a: Sequence[int], b: Sequence[int]) -> Optional[float]:
    """Agreement between two 0/1 label lists, corrected for chance. None if undefined."""
    n = len(a)
    if n == 0 or n != len(b):
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1.0:
        return None
    return round((po - pe) / (1 - pe), 4)


def compare_with_golden(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """rows have `correct` (golden, 0/1), `judge_ok` (0/1 or None) and `id`.
    Invalid judge replies are counted, not hidden."""
    valid = [r for r in rows if r.get("judge_ok") is not None]
    gold = [int(r["correct"]) for r in valid]
    auto = [int(r["judge_ok"]) for r in valid]
    return {"questions": len(rows), "judge_invalid_replies": len(rows) - len(valid), "compared": len(valid),
            "agreement": round(sum(g == a for g, a in zip(gold, auto)) / len(valid), 4) if valid else None,
            "kappa": cohens_kappa(gold, auto),
            "judge_says_ok_golden_says_wrong": [r["id"] for r in valid if r["judge_ok"] and not r["correct"]],
            "judge_says_wrong_golden_says_ok": [r["id"] for r in valid if not r["judge_ok"] and r["correct"]]}
