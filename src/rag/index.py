"""Retrieval: BM25 (keywords) + dense vectors (meaning), metadata filters, rank fusion.

The dense index here is plain numpy, which is enough for the tests and for a corpus
of a few thousand chunks. `Embedder` is the seam: tests use `HashEmbedder`; the lab
plugs in a real sentence-embedding model without changing anything else.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

import numpy as np

from .elements import Element, Hit, Modality

_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenise(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def matches(element: Element, filters: Optional[Dict[str, Any]], modalities: Optional[Iterable[Modality]]) -> bool:
    """Metadata filter. A filter value may be a single value or a list/set of allowed values."""
    if modalities is not None and element.modality not in set(modalities):
        return False
    for key, want in (filters or {}).items():
        have = element.metadata.get(key)
        allowed = want if isinstance(want, (list, set, tuple, frozenset)) else [want]
        if have not in allowed:
            return False
    return True


class BM25Index:
    """Okapi BM25 over element text."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.elements: List[Element] = []
        self._tf: List[Counter] = []
        self._df: Counter = Counter()
        self._len: List[int] = []

    def add(self, elements: Sequence[Element]) -> None:
        for el in elements:
            toks = tokenise(el.text)
            self.elements.append(el)
            self._tf.append(Counter(toks))
            self._len.append(len(toks))
            self._df.update(set(toks))

    def search(self, query: str, k: int = 5, filters=None, modalities=None) -> List[Hit]:
        n = len(self.elements)
        if n == 0:
            return []
        avg = (sum(self._len) / n) or 1.0
        q = tokenise(query)
        scored = []
        for i, el in enumerate(self.elements):
            if not matches(el, filters, modalities):
                continue
            score = 0.0
            for term in q:
                f = self._tf[i].get(term, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                score += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self._len[i] / avg))
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda x: (-x[0], x[1]))  # ties broken by insertion order: deterministic
        return [Hit(self.elements[i], s, r) for r, (s, i) in enumerate(scored[:k], start=1)]


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim), rows L2-normalised."""
        ...


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder for tests. Not a real model."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in tokenise(text):
                h = int.from_bytes(__import__("hashlib").md5(tok.encode()).digest()[:4], "little")
                out[row, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


class DenseIndex:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.elements: List[Element] = []
        self._matrix: Optional[np.ndarray] = None

    def add(self, elements: Sequence[Element]) -> None:
        if not elements:
            return
        vecs = self.embedder.embed([e.text for e in elements])
        self._matrix = vecs if self._matrix is None else np.vstack([self._matrix, vecs])
        self.elements.extend(elements)

    def search(self, query: str, k: int = 5, filters=None, modalities=None) -> List[Hit]:
        if self._matrix is None:
            return []
        sims = self._matrix @ self.embedder.embed([query])[0]
        order = sorted(range(len(self.elements)), key=lambda i: (-float(sims[i]), i))
        hits: List[Hit] = []
        for i in order:
            if matches(self.elements[i], filters, modalities):
                hits.append(Hit(self.elements[i], float(sims[i]), len(hits) + 1))
                if len(hits) == k:
                    break
        return hits


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Hit]], k: int = 60, top: int = 5) -> List[Hit]:
    """Merge ranked lists: score = sum of 1 / (k + rank). Needs no score calibration between lists."""
    fused: Dict[str, float] = {}
    by_id: Dict[str, Element] = {}
    for ranking in rankings:
        for h in ranking:
            eid = h.element.element_id
            by_id[eid] = h.element
            fused[eid] = fused.get(eid, 0.0) + 1.0 / (k + h.rank)
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return [Hit(by_id[eid], score, r) for r, (eid, score) in enumerate(ordered[:top], start=1)]


class HybridIndex:
    """BM25 + dense, fused with reciprocal rank fusion. Text, tables and image text share one index,
    so one query ranks all three modalities together; `modalities` narrows it when needed."""

    def __init__(self, embedder: Embedder, per_list: int = 20):
        self.bm25 = BM25Index()
        self.dense = DenseIndex(embedder)
        self.per_list = per_list

    def add(self, elements: Sequence[Element]) -> None:
        self.bm25.add(elements)
        self.dense.add(elements)

    def __len__(self) -> int:
        return len(self.bm25.elements)

    def save(self, directory: str) -> None:
        save_index(self, directory)

    @classmethod
    def load(cls, directory: str, embedder: Embedder, per_list: int = 20) -> "HybridIndex":
        return load_index(directory, embedder, per_list)

    def search(self, query: str, k: int = 5, filters=None, modalities=None, mode: str = "hybrid") -> List[Hit]:
        if mode not in ("hybrid", "bm25", "dense"):
            raise ValueError(f"mode must be hybrid, bm25 or dense, got {mode!r}")
        if mode == "bm25":   # single list: return it directly so E3 reports true BM25 scores
            return self.bm25.search(query, k, filters, modalities)
        if mode == "dense":  # true cosine scores, not 1/(60+rank)
            return self.dense.search(query, k, filters, modalities)
        lists = []
        if mode in ("hybrid", "bm25"):
            lists.append(self.bm25.search(query, self.per_list, filters, modalities))
        if mode in ("hybrid", "dense"):
            lists.append(self.dense.search(query, self.per_list, filters, modalities))
        return reciprocal_rank_fusion(lists, top=k)


def _element_to_dict(e: Element) -> Dict[str, Any]:
    d = asdict(e)
    d["modality"] = e.modality.value
    return d


def _element_from_dict(d: Dict[str, Any]) -> Element:
    return Element(d["doc_id"], d["page"], Modality(d["modality"]), d["text"],
                   image_path=d.get("image_path"), metadata=d.get("metadata") or {})


def save_index(index: "HybridIndex", directory: str) -> None:
    """Write elements.json and dense.npy. BM25 statistics are rebuilt on load (cheap, deterministic)."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    elements = index.bm25.elements
    (out / "elements.json").write_text(
        json.dumps([_element_to_dict(e) for e in elements], ensure_ascii=False), encoding="utf-8")
    matrix = index.dense._matrix
    if matrix is not None:
        np.save(out / "dense.npy", matrix)


def load_index(directory: str, embedder: Embedder, per_list: int = 20) -> "HybridIndex":
    """Rebuild a HybridIndex from save_index output without re-embedding the documents.
    The embedder is only used to embed queries, so it must be the same model that built the index."""
    src = Path(directory)
    elements = [_element_from_dict(d) for d in json.loads((src / "elements.json").read_text(encoding="utf-8"))]
    idx = HybridIndex(embedder, per_list)
    idx.bm25.add(elements)
    idx.dense.elements = list(elements)
    dense_file = src / "dense.npy"
    if dense_file.exists():
        idx.dense._matrix = np.load(dense_file)
        if len(idx.dense._matrix) != len(elements):
            raise ValueError("dense.npy and elements.json have different lengths: index is corrupt")
    return idx
