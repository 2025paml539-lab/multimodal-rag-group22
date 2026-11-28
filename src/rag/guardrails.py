"""Input and output guardrails and a DoS limiter (brief A5, FR4, E9).

Rules and regular expressions only: no LLM call, so the whole path costs milliseconds and stays far under the
6 s budget (FR4). Every verdict names a category, so the E9 test table can count each one.
The scope check is a coarse keyword test. Its limits are listed in docs/guardrails.md.
"""
from __future__ import annotations
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Optional, Sequence
from .elements import Hit
from .eval_metrics import _content_tokens, cites_retrieved, faithfulness, is_refusal, normalise, parse_citations
from .prompts import REFUSAL

MAX_QUESTION_CHARS = 1000

_INJECTION = [re.compile(p, re.I) for p in (
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (all |any |the )?(previous|prior|above|earlier|your) (instructions|prompts?|rules)",
    r"(reveal|show|print|repeat|output) (me )?(your|the) (system|hidden|initial) (prompt|instructions)",
    r"you are now\b", r"\bact as (a |an )?(?!bank)", r"pretend (to be|you are)",
    r"\b(jailbreak|developer mode|dan mode)\b", r"forget (everything|all|your) (you|instructions|rules)",
    r"override (your|the) (rules|instructions|safety)", r"<\s*/?\s*system\s*>", r"^\s*system\s*:",
)]
_PII = {
    "aadhaar": re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)"),
    "pan": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    "card_number": re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone": re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"),
    "ifsc": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
}
_DOMAIN = tuple("""bank banks banking rbi reserve loan loans credit debit card cards upi payment payments fraud kyc aml
account accounts deposit deposits interest rate rates npa asset assets liability liabilities regulat direction directions
circular master ombudsman complaint customer customers branch nbfc lending borrower repay emi mortgage insurance forex
fema foreign exchange monetary inflation repo capital provision provisioning priority sector fintech wallet transaction
transactions unauthorised unauthorized liability digital cheque neft rtgs imps ifsc atm quarter quarterly chart table
slide deck policy guideline guidelines report document page limit amount maximum minimum score""".split())


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    category: str       # ok | empty | oversized | prompt_injection | personal_data | out_of_scope | ...
    reason: str = ""


def keyword_in_scope(question: str) -> bool:
    """True if the question contains a banking or finance word. Coarse on its own: on the 30 golden questions it
    wrongly rejected 5 (for example "review go to the Board"), so the pipeline also asks retrieval."""
    low = (question or "").lower()
    return any(w in low for w in _DOMAIN)


def check_input(question: str, has_history: bool = False, max_chars: int = MAX_QUESTION_CHARS,
                scope: bool = True) -> Verdict:
    """Block empty, oversized, injection, personal-data and out-of-scope questions.
    With conversation history a short follow-up ('and for the vehicle loan?') skips the scope test."""
    q = question or ""
    if not q.strip():
        return Verdict(False, "empty", "empty question")
    if len(q) > max_chars:
        return Verdict(False, "oversized", f"question has {len(q)} characters, limit is {max_chars}")
    for p in _INJECTION:
        if p.search(q):
            return Verdict(False, "prompt_injection", f"matches pattern {p.pattern[:40]!r}")
    for name, p in _PII.items():
        if p.search(q):
            return Verdict(False, "personal_data", f"looks like {name}")
    if scope and not has_history and not keyword_in_scope(q):
        return Verdict(False, "out_of_scope", "no Indian banking or finance term found")
    return Verdict(True, "ok")


def hits_relevant(question: str, hits: Sequence[Hit], min_overlap: float = 0.15) -> bool:
    """True if enough question words appear in the retrieved text."""
    q = set(_content_tokens(question))
    if not q or not hits:
        return False
    evidence = set(_content_tokens(" ".join(h.element.text for h in hits)))
    return len(q & evidence) / len(q) >= min_overlap


def check_output(answer: str, hits: Sequence[Hit], question: str = "", min_faithful: float = 0.5) -> Verdict:
    """An answer must be grounded and cited. With no relevant evidence the only valid answer is the refusal."""
    if not (answer or "").strip():
        return Verdict(False, "empty_answer", "model returned nothing")
    refused = is_refusal(answer)
    relevant = hits_relevant(question, hits) if question else bool(hits)
    if not hits or not relevant:
        return Verdict(True, "correct_refusal") if refused else Verdict(False, "should_refuse", "no relevant evidence retrieved")
    if refused:
        return Verdict(True, "refusal")
    cites = parse_citations(answer)
    if not cites:
        return Verdict(False, "missing_citation", "answer has no [document, p.N] citation")
    if not cites_retrieved(answer, hits):
        return Verdict(False, "bad_citation", "cites a document or page that was not retrieved")
    evidence_tokens = set(normalise(" ".join(h.element.text for h in hits)).split())
    missing = [n for n in re.findall(r"\d[\d.]*\d|\d", normalise(answer)) if n not in evidence_tokens]
    if missing:   # a number that is not in the retrieved text is the most damaging kind of unsupported claim
        return Verdict(False, "ungrounded", f"number {missing[0]} is not in the retrieved text")
    if faithfulness(answer, hits, min_overlap=0.6) < min_faithful:
        return Verdict(False, "ungrounded", "answer words are not supported by the retrieved text")
    return Verdict(True, "ok")


def guard_output(answer: str, hits: Sequence[Hit], question: str = "") -> tuple:
    """Return (text to show, verdict). A blocked answer is replaced by the refusal."""
    v = check_output(answer, hits, question)
    return (answer if v.allowed else REFUSAL), v


class RateLimiter:
    """Sliding-window limit per session (DoS protection). `clock` is injectable for tests."""

    def __init__(self, max_requests: int = 10, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic):
        if max_requests < 1 or window_s <= 0:
            raise ValueError("max_requests must be >= 1 and window_s > 0")
        self.max_requests, self.window_s, self.clock = max_requests, window_s, clock
        self._seen: Dict[str, Deque[float]] = defaultdict(deque)

    def allow(self, session_id: str) -> Verdict:
        now = self.clock()
        q = self._seen[session_id]
        while q and now - q[0] >= self.window_s:
            q.popleft()
        if len(q) >= self.max_requests:
            return Verdict(False, "rate_limited", f"more than {self.max_requests} requests in {self.window_s:g} s")
        q.append(now)
        return Verdict(True, "ok")
