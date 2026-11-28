# High-level design (Task A3)

Everything below describes code that exists in this repository. Items not built are listed under "Not built yet".

## Components and data flow
```mermaid
flowchart LR
  subgraph Ingestion["Ingestion path (lab, scripts/ingest_corpus.py)"]
    C[corpus/manifest.csv + doc_metadata.csv] --> R[registry.py: file -> doc_id + metadata]
    P[PDF / PPTX files] --> PA[parsers.py: text, tables, images]
    PA --> OCR[ocr.py: RapidOCR on images]
    R --> M[attach doc_type, topic, sensitivity]
    PA --> M
    OCR --> M
    M --> CH[chunking.py: config A 200w/40, config B 60w/10]
    CH --> EM[embedding.py: all-MiniLM-L6-v2 on GPU]
    EM --> IX[(HybridIndex: elements.json + dense.npy)]
  end
  subgraph Query["Query path (pipeline.answer_with_guardrails)"]
    Q[User question] --> DOS[RateLimiter]
    DOS --> GI[guardrails.check_input]
    GI --> MEM[memory.standalone_question]
    MEM --> RET[HybridIndex.search: BM25 + dense, rank fusion, metadata filters]
    IX --> RET
    RET --> PR[prompts.render_prompt: Jinja2 template + ontology]
    PR --> LLM[llm.py: Ollama or hosted client]
    LLM --> GO[guardrails.guard_output: grounded, cited, else refusal]
    GO --> A[Answer with [document, p.N]]
  end
  subgraph Eval["Evaluation (scripts/run_experiments.py)"]
    G[eval/golden_*.jsonl] --> EV[eval_metrics.py]
    A --> EV
    EV --> RES[results/e1 ... e9]
  end
```

## Decisions
| Area | Decision | Why |
|---|---|---|
| Text, tables, images in one index | Every chunk is an `Element` with `modality` text, table or image. Images are searched through their OCR text. | One query ranks all three. `modalities=` narrows when needed (E3 text-only baseline). |
| Retrieval | BM25 + dense vectors, fused by reciprocal rank fusion. Metadata `filters` on any field. | Keywords catch exact numbers and names, vectors catch meaning. Fusion needs no score calibration. |
| Vector store | numpy matrix saved as `dense.npy` + `elements.json`. | About 170 documents; no server needed; the index is a folder that can be downloaded for a laptop demo. |
| Memory | Last 5 turns, follow-up rewritten into a stand-alone question by the LLM. Nothing persists between sessions. | Short-term memory required by Task A2. No long-term memory is built. |
| Ontology | `ontology.yaml` fields are attached to every chunk at ingestion and shown to the prompt (v2, v5). | Metadata filter and prompt placeholders. |
| LLM | Local Ollama (qwen3:8b, deepseek-r1:14b). gpt-oss-20b through a hosted endpoint only (lab Ollama cannot pull it: HTTP 412). | Lab driver 535 limits Ollama to 0.12.6. |
| Guardrails | Rules and regular expressions, no LLM call. | Keeps the path in milliseconds (FR4). |

## Where it runs
Lab (Kubeflow notebook, A100): ingestion, embeddings, every experiment, every reported number. A laptop: code, offline tests, and a demo against a downloaded index. Streamlit cannot be served through the lab proxy (404 in session S2).

## Not built yet
- Streamlit demo app (the integrated path runs through `scripts/e10_demo.py` and the notebook instead).
- Automatic routing from a question to metadata filters (filters exist as an argument; nothing sets them from the question).
- Long-term memory and a query cache.
- Layout or object detection (YOLO). Only OCR is used for images.
