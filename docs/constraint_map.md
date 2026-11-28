# Constraint map (Task A3, brief section 3.3)

"Measured by" names the file the lab run writes. **No measured value is claimed here.** A result exists only after the lab run.

| ID | Requirement | Design decision | Code | Measured by |
|---|---|---|---|---|
| FR1 | Ingest at least 100 PDFs and PPTs | 130 RBI Master Directions approved on 21 Nov, plus 18 extra RBI documents (pending approval) and 20 synthetic decks. Resumable ingest. | `corpus/manifest.csv`, `scripts/ingest_corpus.py` | `results/e1_inventory.json`, `~/rag_index/ingest_summary.json` |
| FR2 | Text, tables and images extracted and searchable | One index of `Element`s with three modalities. Tables as `" | "` rows with header repeated per chunk. Images through OCR text. | `parsers.py`, `ocr.py`, `index.py` | one worked example per modality from `results/final_30/e5_answers.csv`; `results/final_30/e3_retrieval.csv` |
| FR3 | P95 end-to-end latency under 90 s | Local model on the A100, top-5 chunks only, no re-ranking model. | `pipeline.py` | `results/final_30/e6_benchmark.json` (`latency_p95_s`) |
| FR4 | DoS and guardrails add under 6 s | Sliding-window rate limit and rule-based checks, no LLM call. | `guardrails.py` | `results/final_30/e6_benchmark.json` (`guard` block) |
| FR5 | Prompts are self-improving | Critique-and-rewrite loop: the LLM proposes one rule from dev failures, kept only if dev correctness rises; test scored at the end. | `run_experiments.py` (E8) | `results/final_30/e8_iterations.md` |
| NFR1 | Same query repeated 5 times gives the same result | Temperature 0 and a fixed seed; retrieval ties broken by insertion order, so it is deterministic. | `llm.py`, `index.py` | `results/final_30/e6_benchmark.json` (`consistency_identical_answers`) |
| NFR2 | Token use optimised against a naive baseline | Top-k chunks and OCR text instead of whole pages. Naive baseline = every parsed element on the pages of the retrieved hits. | `run_experiments.py` (E6) | `results/final_30/e6_benchmark.json` (`prompt_words_*`, `token_reduction_pct`; counted in words, not model tokens) |

Measured in the lab (`results/final_30/`, 30 golden questions, qwen3:8b on an A100): P95 latency 1.042 s (FR3, limit 90 s); guardrails and DoS P95 4.9 ms (FR4, limit 6 s); prompts self-improving (FR5): the loop ran 3 rounds and no rule helped, reported as measured; consistency (NFR1): retrieved items identical 100 percent, answers identical 90 percent; tokens (NFR2): 39.9 percent fewer words than whole pages (words, not model tokens).

Gaps still open: only 2 of 30 golden questions are image-based, OCR covers the decks and the first documents only, one model (qwen3:8b), answer consistency is below 100 percent, and the guardrail cases were written by us.
