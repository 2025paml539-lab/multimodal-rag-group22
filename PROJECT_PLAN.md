# Project plan — Part A: Multimodal Document Retrieval (Group 22)

PCAM* ZG516 Applied Generative AI · Assignment 2, Part A (12 marks, group)
**Deadline: Monday 30 November 2026, 11:59 PM.** Plan written Saturday 21 November.
Instructor approval of the corpus and scenario: **received** (Prof. Dhruba Adhikary, email of 21 Nov, 9:56 PM, "Approved.").

---

## 0. Status and next actions (updated Sunday 22 November, evening)

### Done

| Item | Who | Where to see it |
|---|---|---|
| Instructor approval of corpus and scenario | Kishore | email of 21 Nov, 9:56 PM; Q/A forum post |
| Lab notebook: A100 GPU, PyTorch 2.8.0+cu128, Ollama 0.12.6 on the GPU (121 tokens/s with qwen3:8b), repos cloned in the lab | Kishore | `docs/LAB_SETUP.md` |
| Repository, project plan, lab guide, `scripts/lab_start.sh` (script not yet run) | Kishore | this repo |
| Core code skeleton, `src/rag/`: elements, chunking, PDF/PPT parsers, BM25 + dense + fusion retrieval, memory, LLM client. 33 offline tests pass | Kishore | `src/rag/`, `tests/` |
| Plan emailed to the team | Kishore | sent 21 Nov |

Nothing from Vishruth, Satya or Bhavani is in the repository yet, and nobody has confirmed a role.
Not yet run on real files: the PDF and PPT parsers, the embedding model, the lab start script, the model pulls for gpt-oss and DeepSeek.

### Next actions, by person

| Who | Next (due) | After that |
|---|---|---|
| **Kishore** | Read `src/rag/index.py` before the lab. Book session S2. In S2 (Mon 23): `git pull`, `lab_start.sh` with the three models, virtual environment from `requirements-lab.txt`, run the parsers on 3 to 5 real documents | S3 (Tue 24): ingest the whole corpus, real embeddings, two chunking configurations (E1, E2) |
| **Vishruth** | Source list of 100+ public documents (title, URL, type) as `corpus/manifest.csv` (Mon 23). Download and dedupe by SHA-256 (Tue 24) | Inventory counts: documents, pages, tables, images. `ontology.yaml`. Model-serving config and token counts |
| **Satya** | Golden-set file format and metric code (Recall@k, MRR, correctness, faithfulness) (Mon 23). Prompt templates v0, the five variants (Mon 23) | Golden set frozen with a dev/test split (Wed 25). Ablation, automatic evaluation, self-improving loop |
| **Bhavani** | Guardrail skeleton: input checks (injection, out-of-scope, personal data) and output checks (grounded, cited, refusal), DoS limiter, test harness (Mon 23). Streamlit skeleton | HLD diagram and constraint map (Tue 24). Non-functional benchmark (E6), report assembly |
| **Everyone** | Reply to confirm your role. Check how many lab sessions your account has and book one for Mon 23 to Wed 25. Send Kishore your GitHub username and the email on it. Write your 8 golden questions from the real documents once the corpus is downloaded (Tue 24) | Your own lesson learned; take part in the demo |
| **Part B (each member separately)** | Choose your own use case from your own work. Keep your repo private. Do not share golden sets | Registry, schemas, golden set, three-model run |

Working rule: branch per person, merge into `main` through a pull request, offline tests pass before every push
(`python -m unittest discover -s tests -t .`).

---

## 1. The most important fact: this project runs in the lab

> **The real work can only be done in the BITS lab (Kubeflow notebook with an NVIDIA A100 80 GB).
> It cannot be done on a personal laptop within the deadline.**

| | Typical laptop | Lab notebook |
|---|---|---|
| GPU | none, or a small one | NVIDIA A100, 80 GB |
| Memory | 8–16 GB | 14 GB for our notebook, 503 GB on the host |
| One LLM answer | 40–70 s on CPU (measured on one of our laptops) | about 1 s (121 tokens/s measured with qwen3:8b) |
| Needed for Part A | an estimated 500+ LLM calls, plus OCR and embeddings for 100+ documents | fits in a few hours (estimate) |

Laptops are used for **writing code, editing documents and running offline tests** (tests use a fake LLM and need no GPU).
Everything that produces a reported number is run in the lab: ingestion, embeddings, retrieval metrics, ablation,
self-improving loop, latency and consistency. Step-by-step lab instructions are in [`docs/LAB_SETUP.md`](docs/LAB_SETUP.md).

Lab limits we have already met (account of Nandhaluru Kishore; teammates should check their own):

- 10 booked sessions per student, 2 hours each. We assume the GPU can only be used inside a booked session (not yet verified, see section 9).
- Notebook limits: 30 GB disk, 14 GB memory, 6 CPUs, 1 GPU.
- GPU driver 535 (CUDA 12.2). We use **PyTorch 2.8.0+cu128** and **Ollama 0.12.6**. Newer Ollama refuses this driver.
- `/tmp` is wiped when the notebook restarts. Code, results and Ollama live in the home folder and survive.

---

## 2. What we must deliver (rubric → evidence → owner)

| Rubric part (marks) | What the marker looks for | Evidence in this repo | Owner |
|---|---|---|---|
| Anchor: corpus and scenario (1.0) | 100+ documents, text + tables + images, clear user and scenario | `corpus/manifest.csv`, inventory table in report | Vishruth |
| A1 Preprocessing and embedding (2.0) | multimodal parsing, OCR/detection, chunking, metadata, embedding choice | `src/ingest/`, `results/e1_inventory.json`, `results/e2_chunking.csv` | Kishore |
| A2 Vector store, memory, retrieval (2.0) | index and metadata design, memory, hybrid retrieval, fusion of text/table/image | `src/index/`, `src/memory/`, `results/e3_retrieval.csv`, `results/e4_transcript.md` | Kishore |
| A3 HLD and LLD (1.5) | diagrams, interfaces, sequence flow, every constraint mapped to a design decision | `docs/HLD.md`, `docs/LLD.md`, `docs/constraint_map.md` | Bhavani (Kishore writes the design notes) |
| A4 Evaluation (2.0) | 30+ golden questions, Recall@k, MRR, correctness, faithfulness, automatic evaluation compared with golden set, 3+ failure cases | `eval/golden_set.jsonl`, `results/e5_*.csv`, `docs/failure_analysis.md` | Satya |
| A5 Prompting layer (1.5) | 5-way ablation, self-improving loop on dev/test split, guardrails, ontology, benchmark table | `prompts/`, `results/e7_ablation.csv`, `results/e8_iterations.md`, `ontology.yaml`, `docs/guardrails.md` | Satya (prompts, loop), Vishruth (ontology), Bhavani (guardrails) |
| Non-functional evidence (1.5) | consistency (5 runs), P95 latency under 90 s, guardrail + DoS overhead under 6 s, token use vs naive baseline | `results/e6_benchmark.csv` | Bhavani |
| A6 Lessons learned (0.5) | honest group reflection and each member's own lesson | `docs/lessons.md`, contribution statement | everyone |
| Deliverables | report (10–15 pages), notebook, demo (8–10 min), signed contribution statement | `report/`, `notebooks/`, demo video link | Bhavani assembles, all review |

Constraints from the brief we must prove with a measurement: P95 end-to-end under 90 s (FR3); DoS and guardrails under 6 s (FR4);
same query run 5+ times gives the same result (NFR1); fewer tokens than a naive "send the whole page" baseline (NFR2).

---

## 3. Team and roles

Kishore built the lab environment from scratch and wants to learn the core system, so he owns the core pipeline end to end.
The other three each own a bounded module with a clear interface, so they can work and test offline without waiting for him.

| Member | Owns | Rubric areas | What they should be able to explain in the demo |
|---|---|---|---|
| **Nandhaluru Kishore** (2025paml539) | Core pipeline and lab lead: document parsing (text, tables, images, OCR), chunking, embeddings, vector index, hybrid retrieval, fusion across modalities, conversation memory, LLM client, integration. Runs the heavy lab jobs. | A1, A2, integration, E1–E4, E10 | how a PDF page becomes chunks and vectors, why hybrid retrieval, how modalities are merged |
| **Vishruth M** (2025paml536) | Corpus and data layer: collect 100+ public documents, dedupe by hash, inventory (documents, pages, tables, images), metadata ontology, model serving config and token counting, E2 configurations | Anchor, ontology part of A5, E1, E2 | why this corpus, how many tables/images, how the ontology narrows retrieval |
| **Satya Prakash** (2025paml561) | Evaluation and prompting: golden set (coordinates the 30+ questions), metrics, automatic evaluation, failure analysis, five prompt variants, self-improving loop, token benchmark | A4, A5, E5, E7, E8 | how quality is measured, what each prompt variant costs, what improved after each iteration |
| **Takkars Bhavani** (2025paml562) | Safety, tests and delivery: input and output guardrails, DoS limiter, non-functional tests (E6), unit tests, Streamlit demo app, HLD/LLD diagrams, report assembly, demo deck, contribution statement | A3, E6, E9, delivery | how an injection attempt is blocked, the latency and consistency numbers, the architecture diagram |
| **Everyone** | write at least 8 golden questions each (text, table and image) so the whole group knows the corpus; each writes their own lesson learned; all four speak in the demo | A6 | the full workflow (the brief says every member must be able to explain it) |

### Module interfaces (agree these on day 1 so work can run in parallel)

- `corpus/manifest.csv`: `doc_id, title, url, type(pdf|ppt), pages, source, sha256, license_note` (Vishruth)
- `ontology.yaml`: document type, topic, section, modality, source, page, date, sensitivity (Vishruth)
- `Element`: `doc_id, page, modality(text|table|image), text, image_path|None, metadata` (Kishore)
- `index.search(query, filters, k) -> list[Hit]`, `Hit = Element + score + source citation` (Kishore)
- `LLMClient.generate(prompt, params) -> str` with a `FakeLLMClient` for offline tests (Kishore)
- `prompts/*.j2` templates take `question, hits, ontology` and return a grounded answer with citations (Satya)
- `guardrails.check_input(q)` and `guardrails.check_output(answer, hits)` (Bhavani)
- `eval/golden_set.jsonl`: `id, question, modality, answer, doc_id, page, split(dev|test)` (Satya)

Rules: branch per person, pull request into `main`, run the offline tests before every push, never commit secrets or raw documents.

---

## 4. Part B (individual) is separate

Each member builds their own prompt registry for 8 marks. Those repos are private until after the deadline, and golden sets must never be shared
between members (a copied golden set scores zero). Part B also needs the lab: each member must run Qwen, DeepSeek and gpt-oss-20b on the A100,
so every member needs their own lab sessions anyway (see section 5).

---

## 5. Lab session budget

Kishore's account has 10 sessions. The group plan needs **7 of them, leaving 3 spare**. Each teammate needs **2 of their own sessions**,
mostly for Part B, plus a little Part A work. Each teammate must check how many sessions their own account has.

| # | Session (account) | Date | Goal | Output saved |
|---|---|---|---|---|
| S1 | Kishore | Sat 21 Nov | **Done.** Notebook, GPU test, PyTorch 2.8.0+cu128, Ollama 0.12.6 on GPU, repos cloned | notes in `docs/LAB_SETUP.md` |
| S2 | Kishore | Mon 23 Nov | Project packages in a virtual environment, pull the three Part B models, smoke test on 5 sample documents, check whether Streamlit is reachable | package list |
| S3 | Kishore | Tue 24 Nov | Full ingestion of the corpus, OCR, tables, images, two chunk/embedding configurations (E1, E2) | index + inventory |
| S4 | Kishore | Wed 25 Nov | Retrieval experiments, text-only vs multimodal (E3), memory demo (E4) | retrieval tables |
| S5 | Kishore | Thu 26 Nov | Answer the golden set, automatic evaluation, prompt ablation (E5, E7) | ablation table |
| S6 | Kishore | Fri 27 Nov | Self-improving loop (E8), guardrail tests (E9), non-functional benchmark (E6) | iteration log, benchmark |
| S7 | Kishore | Sat 28 Nov | Final integrated run (E10) and demo recording | demo video |
| S8–S10 | Kishore | spare | Re-runs after fixes, demo retake | — |
| T-A | each teammate | Mon 23 – Wed 25 | Follow `docs/LAB_SETUP.md` (about 90 min), pull the three models, Part B smoke test | — |
| T-B | each teammate | Thu 26 – Sat 28 | Part B full three-model evaluation; run their Part A piece against the shared index | results pushed or downloaded |

Rules for every session: start with `scripts/lab_start.sh`, end with the checklist in `docs/LAB_SETUP.md` (push or download results, keep a copy outside the lab).
Heavy artifacts (index, extracted images) are shared through Google Drive, code through Git.

---

## 6. Timeline

| Day | Date | Kishore (core, lab) | Vishruth (corpus) | Satya (eval, prompts) | Bhavani (safety, tests, delivery) |
|---|---|---|---|---|---|
| 1 | Sun 22 | Repo structure, module interfaces, parser skeleton with offline tests | Source list for 100+ public documents, collector script | Golden-set template, question categories, metric code | Guardrail skeleton, test harness, Streamlit skeleton |
| 2 | Mon 23 | **S2 in lab**, smoke test | Download and dedupe the corpus, hash list, manifest | Prompt templates v0 (five variants) | Input/output guardrail rules, DoS limiter |
| 3 | Tue 24 | **S3 in lab**: ingestion, embeddings | Inventory counts, ontology file, E2 configs | Write golden questions (8 each from everyone) | HLD diagram, constraint map |
| 4 | Wed 25 | **S4 in lab**: retrieval, memory | Review retrieval examples per modality | Golden set frozen (30+, dev/test split) | Unit tests for retrieval and guardrails |
| 5 | Thu 26 | **S5 in lab**: answers, ablation | Token counts, naive baseline | Run ablation, automatic evaluation | Streamlit app on real index |
| 6 | Fri 27 | **S6 in lab**: self-improvement, guardrails, benchmark | Corpus section of report | Failure analysis (3+ cases) | Non-functional tables, report assembly |
| 7 | Sat 28 | **S7 in lab**: final run, record demo | Rehearsal | Rehearsal | Demo deck, report draft complete |
| 8 | Sun 29 | Fixes, spare session if needed | Proofread | Proofread | Contribution statement, lessons from all four, make repos public |
| 9 | Mon 30 | **Submit by 6 PM**, buffer until 11:59 PM | | | |

Part B runs alongside: each member finishes the registry code and golden set by Fri 27 and the three-model run by Sat 28.

---

## 7. Demo plan (8–10 minutes, all four speak)

Format to confirm: Assignment 1 used a recorded video. We record on Sat 28 in lab session S7 over a Teams or Meet call
with all four joined, Kishore sharing the lab browser. If the Streamlit app is not reachable from the lab we demo from the notebook.

| Time | Speaker | Content |
|---|---|---|
| 0:00–1:00 | Vishruth | Problem, user, corpus: counts of documents, pages, tables, images, ontology |
| 1:00–4:00 | Kishore | Architecture (HLD), then live questions: one text, one table, one image answer, each with its source page; a follow-up question showing memory |
| 4:00–6:30 | Satya | Golden set and metrics, prompt ablation table, self-improvement iterations, token savings against the naive baseline |
| 6:30–9:00 | Bhavani | Guardrail demo (prompt injection, out-of-scope, personal data, refusal when there is no evidence), P50/P95 latency, consistency, DoS overhead |
| 9:00–9:45 | all | One lesson each, in one sentence |

Backup if the lab is down at recording time: pre-recorded clips of the notebook outputs and saved screenshots, taken during S6.

---

## 8. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Lab slot unavailable or a session lost | 3 spare sessions on Kishore's account; teammates' sessions for reruns |
| Everything depends on one account (Kishore) | teammates run T-B on shared artifacts; ask whether teammates can be added as contributors to one namespace (Kubeflow "Manage Contributors") |
| `/tmp` wiped, packages lost | `scripts/lab_start.sh` re-pulls models; packages live in a virtual environment in the home folder |
| Newer software needs a newer driver | pin PyTorch 2.8.0+cu128 and Ollama 0.12.6; never upgrade in the lab |
| OCR or detection packages fail to install | start with the simplest working OCR; YOLO is optional |
| P95 latency over 90 s | smaller model for answers, cache repeated context, limit retrieved chunks |
| Disk (30 GB) fills | models in `/tmp`, index and images compressed, raw documents kept outside the repo |
| Docs download blocked | other public sources from the same domain; keep the source list long |

---

## 9. Things that are easy to miss

- Keep the approval email as evidence and mention it in the report.
- Report the corpus honestly: documents, pages or slides, tables, images. No duplicated files (dedupe by SHA-256).
- Every answer must cite document and page, and refuse when there is no evidence.
- Optimise prompts on the **dev** split only and report on the **test** split. Never tune on the questions used for the final numbers.
- Latency needs many queries (all golden-set queries), not one run. Report P50 and P95. Consistency needs 5 or more runs per query.
- Compare token use against a naive baseline (whole pages sent to the model) with a measurement.
- Automatic evaluation (LLM-as-judge or synthetic questions) must be compared with the golden set, including where it disagrees. Part A allows any model for the judge, but we use the same open-weight models.
- Worked example per modality (text, table, image) with its source.
- HLD and LLD must map every constraint (FR1–FR5, NFR1–NFR2) to a design decision.
- Individual contribution statement signed by all four, each with their own lesson learned.
- Commit no API keys, no raw documents, no passwords. Public documents only.
- After every lab session, copy results out of the lab. The notebook is the only copy otherwise.
- Not yet verified: whether the GPU and notebook work outside a booked slot; whether Streamlit can be reached from the lab; whether two sessions can be booked in one day.
- Submission is on Taxila, Assignment 2: report PDF, notebook, repository link, demo link, contribution statement, golden set, ontology and benchmark files. One member submits for the group. Part B is submitted separately by each member.

## 10. Submission checklist (Mon 30 Nov)

- [ ] Report PDF (10–15 pages excluding appendix)
- [ ] Notebook with stored outputs
- [ ] Repository public, README correct, links work without login
- [ ] Demo video link opens without login
- [ ] Golden set, ontology file, guardrail spec, ablation table, iteration log, benchmark table
- [ ] Contribution statement confirmed by all four members
- [ ] No secrets or confidential data in the repo
- [ ] Each member has submitted their own Part B
