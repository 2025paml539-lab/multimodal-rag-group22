"""The unit of content: one piece of a document page, in one modality."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class Modality(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


@dataclass(frozen=True)
class Element:
    """One chunk of content from one page or slide.

    For an image, `text` holds what we could read from it (OCR text or a caption),
    and `image_path` points at the extracted picture. Tables are stored as text rows
    separated by " | " so they can be searched like text.
    """

    doc_id: str
    page: int                      # 1-based page or slide number
    modality: Modality
    text: str
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("doc_id is required")
        if self.page < 1:
            raise ValueError(f"page must be >= 1, got {self.page}")
        if not isinstance(self.modality, Modality):
            raise ValueError("modality must be a Modality")
        if not self.text.strip() and not self.image_path:
            raise ValueError("an element needs text or an image_path")

    @property
    def element_id(self) -> str:
        return (f"{self.doc_id}|p{self.page}|{self.modality.value}|"
                f"{self.metadata.get('el', 0)}|{self.metadata.get('chunk', 0)}")

    @property
    def citation(self) -> str:
        return f"{self.doc_id}, p.{self.page}"


@dataclass(frozen=True)
class Hit:
    """A retrieved element with its score. `rank` starts at 1."""

    element: Element
    score: float
    rank: int

    @property
    def citation(self) -> str:
        return self.element.citation
