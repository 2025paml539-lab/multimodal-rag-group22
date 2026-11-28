# Multimodal Document Retrieval System — Group 22

**PCAM\* ZG516 Applied Generative AI · Group Major Project / Assignment 2 · Part A (12 marks)**
Deadline: Monday 30 November 2026, 11:59 PM

| Member | BITS ID | Main area |
|---|---|---|
| Nandhaluru Kishore | 2025paml539 | Core pipeline: parsing, embeddings, vector index, retrieval, memory, lab lead |
| Vishruth M | 2025paml536 | Corpus, inventory, metadata ontology, model serving |
| Satya Prakash | 2025paml561 | Golden set, evaluation, prompt variants, self-improving loop |
| Takkars Bhavani | 2025paml562 | Guardrails, tests, non-functional benchmark, demo app, report |

These are planned roles. The final contribution statement is written and confirmed by each member at the end.

> **Important: this project runs in the BITS lab, not on a laptop.**
> The real runs need the lab's NVIDIA A100 80 GB (OCR, embeddings and hundreds of LLM calls). A laptop is only for
> editing code and running the offline tests. Setup is in [`docs/LAB_SETUP.md`](docs/LAB_SETUP.md).

## What we are building

A system that answers questions from at least 100 public banking PDFs and presentations, using their **text, tables and
images**, and cites the document and page for every answer. It refuses when there is no evidence.

- **Domain:** Indian banking and financial regulation (RBI circulars and master directions, public bank presentations,
  financial-literacy decks).
- **User:** a bank compliance or customer-support analyst who needs a sourced answer from long documents.
- **Why retrieval and not fine-tuning:** the sources are long and change over time, and every answer must be traceable
  to a page. Retrieval gives that without training a model.
- **Approval:** the instructor approved the corpus and scenario by email on 21 Nov 2026.

Targets from the brief: 100+ documents; 30+ golden questions; P95 query latency under 90 s; guardrails and DoS checks
under 6 s; the same answer on 5+ repeated runs; fewer tokens than a naive "send whole pages" baseline.

## Documents

| File | What it is |
|---|---|
| [`PROJECT_PLAN.md`](PROJECT_PLAN.md) | Roles, rubric mapping, timeline, lab session budget, demo plan, risks, submission checklist |
| [`docs/LAB_SETUP.md`](docs/LAB_SETUP.md) | Step-by-step lab setup for every member, with the errors we already met |
| `scripts/lab_start.sh` | Start-of-session script (starts Ollama on the GPU, pulls models). Used in every lab session since S2 |

## Run the offline tests (any laptop, no GPU)

```bash
pip install numpy
python -m unittest discover -s tests -t .
```

## Repository layout (planned)

```
corpus/        manifest.csv: one row per document (title, URL, hash). Raw files are not committed
src/           ingest, index, memory, llm client, guardrails
prompts/       Jinja2 prompt templates
eval/          golden_set.jsonl, metric code
results/       experiment outputs (every reported number comes from a file here)
docs/          LAB_SETUP, HLD, LLD, guardrail spec, failure analysis, lessons
notebooks/     executed notebook with stored outputs
report/        report source and build
scripts/       lab helper scripts
tests/         offline tests (use a fake LLM, no GPU needed)
```

Folders other than `docs/` and `scripts/` are created as the work is done.

## Compute

| Task | Where |
|---|---|
| Write code, edit documents, offline tests | personal laptop |
| Ingestion, embeddings, retrieval metrics, LLM runs, benchmarks, demo | BITS lab: Kubeflow notebook, NVIDIA A100 80 GB |

Lab software that works with the lab's driver 535: PyTorch 2.8.0+cu128, Ollama 0.12.6. Do not upgrade either.

## Status (28 Nov 2026) and owners
Dated history by branch: [`docs/WORK_LOG.md`](docs/WORK_LOG.md). Real lab results: [`results/README.md`](results/README.md).
- [x] Instructor approval of corpus and scenario (Kishore)
- [x] Lab environment (A100, PyTorch 2.8.0+cu128, Ollama 0.12.6) and the lab runbook (Kishore)
- [x] Core code `src/rag/`, 90+ offline tests (Kishore)
- [x] Parsers on real documents, OCR, embeddings, saved index: 166 of 167 files, 20,339 chunks (Kishore)
- [x] Corpus manifest, ontology, metadata, inventory (Vishruth)
- [x] 30 golden questions, dev and test split (all four members; checked against the PDF pages)
- [x] Metrics, 5-style prompt ablation, automatic evaluation, self-improving loop: E5, E5B, E7, E8 (code by Kishore in Satya's area)
- [x] Guardrails, DoS limiter, E9; HLD, LLD, constraint map (code by Kishore in Bhavani's area)
- [x] Non-functional benchmark E6, retrieval E3, chunking E2, memory E4: real lab runs in `results/final_30/`
- [ ] E10 integrated demonstration and executed notebook (run in the lab)
- [ ] Demo app, 8 to 10 minute video, Part A report (draft in `report/REPORT_DRAFT.md`), lessons learned, signed contribution statement
- [ ] Part B (each member, own repository)

## Rules we follow

- No confidential, proprietary or personal data. Public documents only.
- Branch per person, pull request into `main`, offline tests pass before every push.
- No secrets in the repository. Raw documents and model files are not committed.
- Every number in the report comes from a result file in this repository.
