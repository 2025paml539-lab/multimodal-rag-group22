"""End-to-end smoke test: documents -> parse -> chunk -> embed -> hybrid search -> grounded answer -> memory follow-up.

Not an experiment. It checks that every stage works together before the real corpus arrives (lab session S3).

Offline (laptop, no GPU, no model):
    python scripts/e2e_smoke.py --docs smoke_docs --embedder hash --llm fake
Lab (real embeddings on the GPU, real LLM through Ollama):
    python scripts/e2e_smoke.py --docs ~/smoke_docs --embedder st --llm ollama --model qwen3:8b
"""
from __future__ import annotations

import argparse
import collections
import glob
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from rag import (ChunkConfig, ConversationMemory, FakeLLMClient, HashEmbedder, HybridIndex,  # noqa: E402
                 OllamaClient, chunk_elements, count_tokens)
from rag import parsers  # noqa: E402

QUESTIONS = [  # (kind, question) written for the 4 sample documents made for this test
    ("text", "What is the minimum credit score for unsecured personal loans?"),
    ("table", "What is the maximum amount for a home loan?"),
    ("image", "How many fraud cases were reported in the third quarter?"),
    ("table", "Which payment channel lost the most money to fraud?"),
]
FOLLOW_UP = "And what about the vehicle loan?"

PROMPT = """Answer the question using only the context below. Cite the source like [document, p.N].
If the answer is not in the context, reply exactly: Not found in the documents.

Context:
{context}

Question: {question}
Answer:"""


class SentenceTransformerEmbedder:
    """Real embedding model on the GPU. Rows are L2-normalised, as HybridIndex expects."""

    def __init__(self, name: str):
        from sentence_transformers import SentenceTransformer
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(name, device=self.device)
        self.name = name

    def embed(self, texts):
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True,
                                            convert_to_numpy=True, batch_size=64), dtype=np.float32)


def load_ocr():
    try:
        from rapidocr_onnxruntime import RapidOCR
        engine = RapidOCR()

        def ocr(path: str) -> str:
            res, _ = engine(path)
            return " ".join(r[1] for r in res) if res else ""
        return ocr
    except Exception as e:  # OCR is optional for this test
        print("  OCR not available:", e)
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="smoke_docs")
    ap.add_argument("--embedder", choices=["hash", "st"], default="hash")
    ap.add_argument("--st-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--llm", choices=["none", "fake", "ollama"], default="none")
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--images", default=os.path.expanduser("~/smoke_images"))
    a = ap.parse_args()

    t0 = time.perf_counter()
    files = sorted(f for f in glob.glob(os.path.join(os.path.expanduser(a.docs), "*")) if f.lower().endswith((".pdf", ".pptx")))
    if not files:
        print("No PDF or PPTX files found in", a.docs)
        return 1

    print("1. PARSE")
    ocr, elements = load_ocr(), []
    for f in files:
        doc_id = Path(f).stem
        els = parsers.parse_document(f, doc_id, os.path.join(a.images, doc_id), ocr)
        print(f"  {doc_id}: {dict(collections.Counter(e.modality.name for e in els))}")
        elements += els

    print("2. CHUNK (two configurations, like experiment E2)")
    configs = {"A (200 words, overlap 40)": ChunkConfig(200, 40, 8), "B (60 words, overlap 10)": ChunkConfig(60, 10, 2)}
    chunks = {}
    for name, cfg in configs.items():
        chunks[name] = chunk_elements(elements, cfg)
        print(f"  config {name}: {len(chunks[name])} chunks, {sum(count_tokens(c.text) for c in chunks[name])} words")
    use = chunks["A (200 words, overlap 40)"]

    print("3. EMBED + INDEX")
    t = time.perf_counter()
    embedder = HashEmbedder() if a.embedder == "hash" else SentenceTransformerEmbedder(a.st_model)
    if a.embedder == "st":
        print(f"  model {embedder.name} on {embedder.device}")
    index = HybridIndex(embedder)
    index.add(use)
    print(f"  {len(index)} chunks indexed in {time.perf_counter() - t:.1f}s")

    print("4. SEARCH (bm25 / dense / hybrid)")
    for kind, q in QUESTIONS:
        print(f"  [{kind}] {q}")
        for mode in ("bm25", "dense", "hybrid"):
            hits = index.search(q, k=3, mode=mode)
            print(f"    {mode:6s}", " | ".join(f"{h.citation} {h.element.modality.value}" for h in hits))

    if a.llm == "none":
        print("5. LLM skipped (--llm none)")
        print(f"Done in {time.perf_counter() - t0:.1f}s")
        return 0

    llm = FakeLLMClient(lambda p: "Fake answer [fake, p.1]") if a.llm == "fake" else OllamaClient(a.model)
    print(f"5. ANSWER with {a.llm}" + (f" ({a.model})" if a.llm == "ollama" else ""))
    memory = ConversationMemory()
    for kind, q in QUESTIONS:
        hits = index.search(q, k=a.k)
        ctx = "\n".join(f"[{h.citation}] ({h.element.modality.value}) {h.element.text}" for h in hits)
        r = llm.generate(PROMPT.format(context=ctx, question=q))
        cited = any(h.citation in r.text for h in hits)
        print(f"  Q[{kind}]: {q}\n  A: {r.text[:240]}\n     tokens in/out {r.input_tokens}/{r.output_tokens}, {r.seconds:.1f}s, cites a retrieved source: {cited}")
        memory.add(q, r.text)

    print("6. MEMORY follow-up (experiment E4 idea)")
    standalone = memory.standalone_question(FOLLOW_UP, llm)
    print(f"  follow-up : {FOLLOW_UP}\n  rewritten : {standalone}")
    hits = index.search(standalone, k=a.k)
    print("  top hits  :", " | ".join(f"{h.citation} {h.element.modality.value}" for h in hits[:3]))
    print(f"Done in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
