"""One question -> retrieve -> prompt -> LLM -> answer. Shared by the experiment runner and the demo app."""
from __future__ import annotations
import json
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence
from .elements import Hit, Modality
from .index import HybridIndex
from .llm import LLMClient
from .prompts import render_prompt


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
