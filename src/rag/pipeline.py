"""One question -> retrieve -> prompt -> LLM -> answer. Shared by the experiment runner and the demo app."""
from __future__ import annotations
import json
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence
from .elements import Hit, Modality
from .index import HybridIndex
from .llm import LLMClient
from .guardrails import RateLimiter, check_input, guard_output, hits_relevant, keyword_in_scope
from .memory import ConversationMemory
from .prompts import REFUSAL, render_prompt


def normalise_answer(variant: str, text: str) -> str:
    """The JSON variant (v5) is turned into plain text with [document, p.N] citations so one metric code works."""
    if variant != "v5_strict_refusal_schema":
        return text.strip()
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        obj = json.loads(text[start:end])
        cites = " ".join(f"[{c['document']}, p.{c['page']}]" for c in obj.get("citations", []))
        return f"{obj.get('answer', '').strip()} {cites}".strip()
    except Exception:
        return text.strip()


def answer_question(question: str, index: HybridIndex, llm: LLMClient, variant: str = "v2_citation_rules",
                    k: int = 5, mode: str = "hybrid", modalities: Optional[Iterable[Modality]] = None,
                    filters: Optional[Dict[str, Any]] = None, ontology: Optional[dict] = None,
                    max_tokens: int = 256, hits: Optional[Sequence[Hit]] = None,
                    extra_rules: str = "") -> Dict[str, Any]:
    t0 = time.perf_counter()
    if hits is None:
        hits = index.search(question, k=k, filters=filters, modalities=modalities, mode=mode)
    t1 = time.perf_counter()
    prompt = render_prompt(variant, question, hits, ontology)
    if extra_rules.strip():  # rules added by the self-improving loop (E8)
        prompt = f"Additional rules:\n{extra_rules.strip()}\n\n{prompt}"
    res = llm.generate(prompt, temperature=0.0, max_tokens=max_tokens)
    t2 = time.perf_counter()
    return {"answer": normalise_answer(variant, res.text), "raw": res.text, "hits": list(hits), "prompt": prompt,
            "prompt_words": len(prompt.split()), "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens, "retrieve_s": t1 - t0, "total_s": t2 - t0}


def answer_with_guardrails(question: str, index: HybridIndex, llm: LLMClient, limiter: RateLimiter,
                           session_id: str = "default", memory: Optional[ConversationMemory] = None,
                           variant: str = "v2_citation_rules", **kwargs: Any) -> Dict[str, Any]:
    """The full query path of the brief (3.6): DoS and input guardrail -> memory rewrite -> retrieval -> prompt
    -> LLM -> grounded, cited answer -> output guardrail. A blocked step returns the reason and the refusal text."""
    t0 = time.perf_counter()
    gate = limiter.allow(session_id)
    if not gate.allowed:
        return {"text": REFUSAL, "stage": "dos", "verdict": gate, "guard_s": time.perf_counter() - t0}
    verdict = check_input(question, has_history=bool(memory and len(memory)), scope=False)
    if not verdict.allowed:
        return {"text": REFUSAL, "stage": "input", "verdict": verdict, "guard_s": time.perf_counter() - t0}
    guard_in = time.perf_counter() - t0
    standalone = memory.standalone_question(question, llm) if memory else question
    hits = index.search(standalone, k=kwargs.get("k", 5), filters=kwargs.get("filters"),
                        modalities=kwargs.get("modalities"), mode=kwargs.get("mode", "hybrid"))
    if not keyword_in_scope(standalone) and not hits_relevant(standalone, hits):
        # no banking word and nothing relevant in the corpus: out of scope, and no LLM call is made
        from .guardrails import Verdict
        return {"text": REFUSAL, "stage": "input", "verdict": Verdict(False, "out_of_scope", "no banking term and no relevant text"),
                "guard_s": time.perf_counter() - t0}
    res = answer_question(standalone, index, llm, variant, hits=hits, **kwargs)
    t1 = time.perf_counter()
    text, out_verdict = guard_output(res["answer"], res["hits"], standalone)
    guard_s = guard_in + (time.perf_counter() - t1)
    if memory is not None:
        memory.add(question, text)
    return {"text": text, "stage": "output", "verdict": out_verdict, "question_used": standalone,
            "hits": res["hits"], "raw_answer": res["answer"], "input_tokens": res["input_tokens"],
            "output_tokens": res["output_tokens"], "guard_s": guard_s, "total_s": time.perf_counter() - t0}
