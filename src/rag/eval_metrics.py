"""Evaluation metrics (A4) and the golden-set loader. Pure functions: no model needed for tests.

Golden schema (one JSON object per line): id, question, modality, answer, doc_id, page, split.
Files are eval/golden_*.jsonl, one per member; load_golden() merges them.
"""
from __future__ import annotations
import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from .elements import Hit
from .llm import LLMClient

REQUIRED = ("id", "question", "modality", "answer", "doc_id", "page", "split")
SPLITS = ("dev", "test")
MODALITIES = ("text", "table", "image")
REFUSAL = "not found in the documents"


def load_golden(directory: str = "eval", pattern: str = "golden_*.jsonl") -> List[Dict[str, Any]]:
    """Merge every golden file. Fails loudly on a duplicate id, a missing field or a bad split/modality."""
    rows, seen = [], {}
    for path in sorted(glob.glob(str(Path(directory) / pattern))):
        for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            where = f"{path}:{n}"
            missing = [k for k in REQUIRED if k not in row or row[k] in (None, "")]
            if missing:
                raise ValueError(f"{where}: missing {missing}")
            if row["split"] not in SPLITS:
                raise ValueError(f"{where}: split must be one of {SPLITS}, got {row['split']!r}")
            if row["modality"] not in MODALITIES:
                raise ValueError(f"{where}: modality must be one of {MODALITIES}, got {row['modality']!r}")
            if row["id"] in seen:
                raise ValueError(f"{where}: duplicate id {row['id']!r} (first in {seen[row['id']]})")
            seen[row["id"]] = where
            rows.append(row)
    return rows


def recall_at_k(hits: Sequence[Hit], gold_doc: str, gold_page: int, k: int = 5) -> float:
    """1.0 if the gold document+page is among the top k hits, else 0.0."""
    return 1.0 if any(h.element.doc_id == gold_doc and h.element.page == gold_page for h in hits[:k]) else 0.0


def mrr(hits: Sequence[Hit], gold_doc: str, gold_page: int) -> float:
    """Reciprocal rank of the first hit on the gold document+page (0.0 if none)."""
    for i, h in enumerate(hits, 1):
        if h.element.doc_id == gold_doc and h.element.page == gold_page:
            return 1.0 / i
    return 0.0


_CITE = re.compile(r"\[([^\[\],]+),\s*p\.?\s*(\d+)\]")


def parse_citations(answer: str) -> List[Tuple[str, int]]:
    return [(d.strip(), int(p)) for d, p in _CITE.findall(answer)]


def cites_retrieved(answer: str, hits: Sequence[Hit]) -> bool:
    """True if the answer has at least one citation and every citation is a retrieved document+page."""
    cites = parse_citations(answer)
    have = {(h.element.doc_id, h.element.page) for h in hits}
    return bool(cites) and all(c in have for c in cites)


_NUM_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
_NUM_WORDS.update({"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
                   "ninety": "90"})
_CURRENCY = re.compile(r"₹|\brs\.?(?=\s|\d)|\binr\b|\brupees?\b")


def normalise(text: str) -> str:
    """Lower-case, drop citations and punctuation. Equivalent forms are made identical so that a correct answer is
    not marked wrong for its wording: number words (three = 3) and rupee marks (₹, Rs., INR, rupees = rs).
    Added after lab session S3, where 4 of 5 answers scored wrong were correct but worded differently."""
    t = text.lower()
    t = _CURRENCY.sub(" rs ", t)
    t = re.sub(r"\b(" + "|".join(_NUM_WORDS) + r")\b", lambda m: _NUM_WORDS[m.group(1)], t)
    t = re.sub(r"(?<=\d),(?=\d)", "", t)          # 1,250 -> 1250
    t = _CITE.sub(" ", t)                         # drop citations
    t = re.sub(r"[^a-z0-9.\s]", " ", t)
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)     # keep decimal points, drop sentence dots
    return " ".join(t.split())


def is_refusal(answer: str) -> bool:
    return REFUSAL in normalise(answer)


def correctness(prediction: str, gold: str, judge: Optional[LLMClient] = None) -> float:
    """1.0 if the normalised gold answer appears in the prediction (word boundaries).
    If it does not and a judge LLM is given, the judge decides (yes/no). Refusing a gold answer is 0.0."""
    p, g = normalise(prediction), normalise(gold)
    if g and re.search(rf"(?<![a-z0-9.]){re.escape(g)}(?![a-z0-9]|\.\d)", p):
        return 1.0
    if judge is not None and not is_refusal(prediction):
        prompt = ("Is the candidate answer equivalent to the reference answer? Reply only yes or no.\n"
                  f"Reference: {gold}\nCandidate: {prediction}\nEquivalent:")
        return 1.0 if judge.generate(prompt, max_tokens=3).text.strip().lower().startswith("yes") else 0.0
    return 0.0


_STOP = set("the a an of to in on for and or is are was were be by with as at from that this it its their "
            "which who has have had not no but if then than so such into per".split())


def _content_tokens(text: str) -> List[str]:
    return [t for t in normalise(text).split() if t not in _STOP and (len(t) > 2 or t.isdigit())]


def faithfulness(answer: str, hits: Sequence[Hit], min_overlap: float = 0.8) -> float:
    """Fraction of answer sentences whose content words appear in the retrieved text.
    A correct refusal counts as faithful (1.0). An empty answer is 0.0."""
    if not answer.strip():
        return 0.0
    if is_refusal(answer):
        return 1.0
    evidence = set(_content_tokens(" ".join(h.element.text for h in hits)))
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if _content_tokens(s)]
    if not sentences:
        return 0.0
    ok = 0
    for s in sentences:
        toks = _content_tokens(s)
        if sum(t in evidence for t in toks) / len(toks) >= min_overlap:
            ok += 1
    return ok / len(sentences)


def summarise(rows: Iterable[Dict[str, Any]], keys: Sequence[str] = ("recall", "mrr", "correct", "faithful")) -> Dict[str, Any]:
    """Mean of each metric overall and per modality and per split. Rows are dicts with those keys plus modality, split."""
    rows = list(rows)
    out: Dict[str, Any] = {"n": len(rows)}
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups["all"].append(r)
        groups[f"modality:{r['modality']}"].append(r)
        groups[f"split:{r['split']}"].append(r)
    for g, rs in sorted(groups.items()):
        out[g] = {"n": len(rs), **{k: round(sum(r[k] for r in rs) / len(rs), 4) for k in keys if all(k in r for r in rs)}}
    return out
