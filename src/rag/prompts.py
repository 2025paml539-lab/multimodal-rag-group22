"""Prompt variants for the ablation (E7). Templates live in prompts/*.j2 (Jinja2)."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from .elements import Hit

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
VARIANTS = ("v1_minimal", "v2_citation_rules", "v3_reasoning", "v4_few_shot", "v5_strict_refusal_schema")
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
    if variant not in VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}; choose from {VARIANTS}")
    return _env().get_template(f"{variant}.j2").render(question=question, hits=list(hits), ontology=ontology)


def load_ontology(path: Optional[str] = None) -> Dict[str, Any]:
    import yaml
    p = Path(path) if path else PROMPT_DIR.parent / "ontology.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))
