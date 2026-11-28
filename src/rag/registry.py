"""Map files to manifest doc_ids and attach document-level metadata (ontology.yaml fields)."""
from __future__ import annotations
import csv
from dataclasses import replace
from pathlib import Path
from typing import Dict, List
from .elements import Element

META_FIELDS = ("doc_type", "topic", "sensitivity", "source", "date")


def load_registry(manifest_path: str, metadata_path: str) -> Dict[str, Dict[str, str]]:
    """Return {file stem: {"doc_id": ..., doc_type, topic, ...}}.

    PDFs are saved as <doc_id>.pdf, so the stem is the doc_id. Local files (the synthetic decks, whose url is a
    path such as corpus/sample_decks/SAMPLE_01_x.pptx) are keyed by their file stem and mapped to the manifest doc_id.
    """
    meta = {}
    if Path(metadata_path).exists():
        with open(metadata_path, encoding="utf-8", newline="") as f:
            meta = {r["doc_id"]: r for r in csv.DictReader(f)}
    reg: Dict[str, Dict[str, str]] = {}
    if not Path(manifest_path).exists():
        return reg
    with open(manifest_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            doc_id = row["doc_id"]
            url = (row.get("url") or "").strip()
            stem = doc_id if url.lower().startswith("http") else Path(url).stem
            info = {"doc_id": doc_id, **{k: (meta.get(doc_id, {}).get(k) or "") for k in META_FIELDS}}
            reg[stem] = info
    return reg


def attach_metadata(elements: List[Element], info: Dict[str, str]) -> List[Element]:
    """Copy document-level metadata onto every element (used for filters and prompts)."""
    extra = {k: info[k] for k in META_FIELDS if info.get(k)}
    return [replace(e, doc_id=info["doc_id"], metadata={**e.metadata, **extra}) for e in elements]
