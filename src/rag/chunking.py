"""Chunking for text, tables and image text.

Token counts here are whitespace words, which is a cheap stand-in. The lab runs
report real counts from the model tokenizer (see PROJECT_PLAN.md, E6).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Sequence

from .elements import Element, Modality


def count_tokens(text: str) -> int:
    return len(text.split())


@dataclass(frozen=True)
class ChunkConfig:
    """A chunking configuration. Two or more of these are compared in experiment E2."""

    max_tokens: int = 200
    overlap: int = 40
    table_rows_per_chunk: int = 8

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if not 0 <= self.overlap < self.max_tokens:
            raise ValueError("overlap must be >= 0 and smaller than max_tokens")
        if self.table_rows_per_chunk < 1:
            raise ValueError("table_rows_per_chunk must be >= 1")


def chunk_text(text: str, max_tokens: int = 200, overlap: int = 40) -> List[str]:
    """Sliding word window. Short text comes back as one chunk."""
    ChunkConfig(max_tokens=max_tokens, overlap=overlap)  # validates
    words = text.split()
    if not words:
        return []
    if len(words) <= max_tokens:
        return [" ".join(words)]
    step = max_tokens - overlap
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + max_tokens]))
        if start + max_tokens >= len(words):
            break
        start += step
    return chunks


def chunk_table(rows: Sequence[Sequence[str]], rows_per_chunk: int = 8) -> List[str]:
    """Group table rows, repeating the header row in every chunk so each chunk is readable alone."""
    if rows_per_chunk < 1:
        raise ValueError("rows_per_chunk must be >= 1")
    cleaned = [[(c or "").strip() for c in row] for row in rows if any((c or "").strip() for c in row)]
    if not cleaned:
        return []
    header, body = cleaned[0], cleaned[1:]
    fmt = lambda r: " | ".join(r)
    if not body:
        return [fmt(header)]
    return [
        "\n".join([fmt(header)] + [fmt(r) for r in body[i:i + rows_per_chunk]])
        for i in range(0, len(body), rows_per_chunk)
    ]


def chunk_elements(elements: Sequence[Element], config: ChunkConfig = ChunkConfig()) -> List[Element]:
    """Split parsed elements into chunks, numbering them in metadata['chunk'].

    - text: sliding window
    - table: elements are expected to hold rows joined by newlines, with the header first
    - image: kept whole (the OCR text of one image is short)
    """
    out: List[Element] = []
    for el in elements:
        if el.modality == Modality.TEXT:
            parts = chunk_text(el.text, config.max_tokens, config.overlap)
        elif el.modality == Modality.TABLE:
            rows = [line.split(" | ") for line in el.text.splitlines()]
            parts = chunk_table(rows, config.table_rows_per_chunk)
        else:
            parts = [el.text] if el.text.strip() else [""]
        for i, part in enumerate(parts):
            if not part and not el.image_path:
                continue
            out.append(replace(el, text=part, metadata={**el.metadata, "chunk": i}))
    return out
