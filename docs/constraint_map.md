# Constraint map (Task A3, brief section 3.3)

"Measured by" names the file the lab run writes. **No measured value is claimed here.** A result exists only after the lab run.

| ID | Requirement | Design decision | Code | Measured by |
|---|---|---|---|---|
| FR1 | Ingest at least 100 PDFs and PPTs | 130 RBI Master Directions approved on 21 Nov, plus 18 extra RBI documents (pending approval) and 20 synthetic decks. Resumable ingest. | `corpus/manifest.csv`, `scripts/ingest_corpus.py` | `results/e1_inventory.json`, `~/rag_index/ingest_summary.json` |
| FR2 | Text, tables and images extracted and searchable | One index of `Element`s with three modalities. Tables as `" | "` rows with header repeated per chunk. Images through OCR text. | `parsers.py`, `ocr.py`, `index.py` | one worked example per modality from `results/e5_answers.csv`; `results/e3_retrieval.csv` |
| FR3 | P95 end-to-end latency under 90 s | Local model on the A100, top-5 chunks only, no re-ranking model. | `pipeline.py` | `results/e6_benchmark.json` (`latency_p95_s`) |
| FR4 | DoS and guardrails add under 6 s | Sliding-window rate limit and rule-based checks, no LLM call. | `guardrails.py` | `results/e6_benchmark.json` (`guard` block) |
| FR5 | Prompts are self-improving | Critique-and-rewrite loop: the LLM proposes one rule from dev failures, kept only if dev correctness rises; test scored at the end. | `run_experiments.py` (E8) | `results/e8_iterations.md` |
| NFR1 | Same query repeated 5 times gives the same result | Temperature 0 and a fixed seed; retrieval ties broken by insertion order, so it is deterministic. | `llm.py`, `index.py` | `results/e6_benchmark.json` (`consistency_identical_answers`) |
| NFR2 | Token use optimised against a naive baseline | Top-k chunks and OCR text instead of whole pages. Naive baseline = every parsed element on the pages of the retrieved hits. | `run_experiments.py` (E6) | `results/e6_benchmark.json` (`prompt_words_*`, `token_reduction_pct`; counted in words, not model tokens) |

Gaps against the brief that are still open: the automatic evaluation method and its comparison with the golden set (A4), the prompt ablation using the five styles the brief names (E7 currently uses five Jinja2 variants), 30 or more golden questions (8 exist), and the demo.
