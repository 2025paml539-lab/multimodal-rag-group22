# Lab results guide

All files here were produced by `scripts/run_experiments.py` in the BITS lab (NVIDIA A100 80 GB, driver 535.309.01, qwen3:8b through Ollama 0.12.6, embeddings all-MiniLM-L6-v2). Values were copied from the lab unchanged. Nothing in this folder comes from the fake LLM or from the hash embedder.

| Folder | Questions | Index | What it is |
|---|---|---|---|
| `final_30/` | 30 hand-checked golden questions | with document titles | **The run to report.** E3, E5, E5B, E6, E7, E8. |
| `handwritten_v2/` | first 8 golden questions | with document titles | Same experiments on the 8 questions, also with retrieval consistency in E6. |
| `synthetic/` | 60 synthetic-deck questions | **before** the title fix | Table recall 0.0. This is the failure that led to the fix. |
| `synthetic_v2/` | 60 synthetic-deck questions | with document titles | E3 and E5 only. Table recall 1.0, table correctness 0.95. |
| `s3_strict_match/` | 8 golden questions | before the title fix | First results with the strict answer check (correct 0.375). Kept as the record of the metric problem. |
| top level `e*.csv/json/md` | 8 golden questions | before the title fix | First full run of E1 to E9 with the corrected answer check. |

Environment records: `lab_gpu.txt`, `lab_models.txt`, `lab_packages.txt`, `ingest.log`, `ingest_summary.json`.

## Known limits
- `e2_chunking.csv`, `e1_inventory.json` and `e9_guardrails.csv` in the top level were produced before the title fix and were not rerun. E1 counts do not change with the fix. E2 should be rerun before it is quoted.
- 167 files were found in the lab and 166 parsed: D147 failed with a PyMuPDF error (see `ingest_summary.json`). The lab had 147 of the 148 PDFs in the manifest.
- OCR text exists for the 20 synthetic decks and the first documents only. Later PDFs were ingested without OCR because OCR was slow in the lab.
- The final run has only 2 truly image-based golden questions. The D145 and D146 pages chosen for image questions hold their words in the text layer and are recorded as text or table.
- Answers are identical in 90 percent of repeated runs (final_30), retrieval in 100 percent. The unstable questions are listed in `final_30/e6_benchmark.json`.
