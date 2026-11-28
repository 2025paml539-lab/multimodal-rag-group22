"""Multimodal document retrieval (Group 22, Assignment 2 Part A).

Modules
-------
elements   Element / Hit: the unit of content that is indexed and cited
chunking   split text, tables and image text into chunks
parsers    PDF and PPT parsing into Elements (needs PyMuPDF / python-pptx, used in the lab)
index      BM25 + dense retrieval, metadata filters, reciprocal-rank fusion
memory     short-term conversational memory and follow-up rewriting
llm        LLM client interface, a fake client for tests, Ollama and hosted OpenAI-compatible clients
embedding  SentenceTransformerEmbedder (lab only, lazy import)
ocr        RapidOCR wrapper (lab only, lazy import)

Everything except the real parsers and the Ollama client runs offline with numpy only.
"""

from .elements import Element, Hit, Modality
from .chunking import ChunkConfig, chunk_elements, chunk_table, chunk_text, count_tokens
from .index import BM25Index, DenseIndex, HashEmbedder, HybridIndex, reciprocal_rank_fusion
from .llm import FakeLLMClient, HostedOpenAICompatibleClient, LLMResult, OllamaClient
from .memory import ConversationMemory

__all__ = [
    "Element", "Hit", "Modality",
    "ChunkConfig", "chunk_elements", "chunk_table", "chunk_text", "count_tokens",
    "BM25Index", "DenseIndex", "HashEmbedder", "HybridIndex", "reciprocal_rank_fusion",
    "FakeLLMClient", "HostedOpenAICompatibleClient", "LLMResult", "OllamaClient",
    "ConversationMemory",
]
