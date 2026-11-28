"""Prompt variants for the ablation (E7). Templates live in prompts/*.j2 (Jinja2)."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from .elements import Hit

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
VARIANTS = ("v1_minimal", "v2_citation_rules", "v3_reasoning", "v4_few_shot", "v5_strict_refusal_schema",
            "zero_shot", "fstring", "three_shot", "create")
# The five styles the brief asks to compare in E7. "jinja2" is the Jinja2 template v2_citation_rules.
ABLATION_STYLES = ("zero_shot", "fstring", "three_shot", "jinja2", "create")
_ALIAS = {"jinja2": "v2_citation_rules"}
REFUSAL = "Not found in the documents."


def _env():
    try:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined
    except ImportError as e:
        raise RuntimeError("jinja2 is not installed (pip install jinja2)") from e
    return Environment(loader=FileSystemLoader(str(PROMPT_DIR)), undefined=StrictUndefined,
                       trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)


def render_prompt(variant: str, question: str, hits: Sequence[Hit],
                  ontology: Optional[Dict[str, Any]] = None) -> str:
    """Render one variant. `ontology` is the parsed ontology.yaml dict (optional)."""
    variant = _ALIAS.get(variant, variant)
    if variant not in VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}; choose from {VARIANTS}")
    if variant == "fstring":   # plain Python f-string, no template engine
        context = "\n".join(f"[{h.element.doc_id}, p.{h.element.page}] ({h.element.modality.value})\n{h.element.text}"
                            for h in hits)
        return (f"Answer the question using only the context. Cite the document and page as [document, p.N]. "
                f"If the answer is not in the context, reply exactly: {REFUSAL}\n\nContext:\n{context}\n\n"
                f"Question: {question}\nAnswer:\n")
    return _env().get_template(f"{variant}.j2").render(question=question, hits=list(hits), ontology=ontology)


def load_ontology(path: Optional[str] = None) -> Dict[str, Any]:
    import yaml
    p = Path(path) if path else PROMPT_DIR.parent / "ontology.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))
