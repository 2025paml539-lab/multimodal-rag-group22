# Lab runbook (sessions after S2)

Paste the blocks in order. Each step says which file it produces. `/tmp` is wiped between sessions: models are pulled again by `lab_start.sh`, while the index, caches and results live in the home folder and survive.

## 0. Every session (5 minutes)
```bash
source ~/venv_rag/bin/activate
cd ~/multimodal-rag-group22 && git checkout main && git pull
bash scripts/lab_start.sh qwen3:8b            # expect library=CUDA
```
Raw PDFs are not in git. If `ls data/raw | wc -l` is not 147, upload `raw_pdfs.zip` and run `unzip -oq ~/raw_pdfs.zip -d data/raw`.

## 1. Ingest the corpus (the slowest step, start it first)
```bash
nohup python scripts/ingest_corpus.py --embedder st --ocr > ~/ingest.log 2>&1 &
tail -f ~/ingest.log                            # Ctrl+C stops tail only
```
Produces `~/rag_cache/parsed/*.json` (one file per document, so a restart resumes) and `~/rag_index/` (`elements.json`, `dense.npy`, `ingest_summary.json`). Run a first check with `--limit 5` to measure seconds per document.

## 2. While ingest runs: more models (one per family, Part B and ablations)
```bash
bash scripts/lab_start.sh deepseek-r1:14b       # downloads in the background of your attention
```
gpt-oss-20b: the lab Ollama 0.12.6 cannot pull it (HTTP 412). Fallback is a hosted open-weight endpoint (Part B rule 15). Set the variables in the shell only, never in a file:
```bash
export RAG_LLM_BASE_URL=https://<provider>/v1
export RAG_LLM_MODEL=openai/gpt-oss-20b
read -s RAG_LLM_API_KEY && export RAG_LLM_API_KEY   # type the key, press Enter; nothing is echoed
```
Report provider, quantisation and hardware for this model. If Prof. Adhikary replies otherwise, switch back to a local pull.

## 3. Run the experiments (after ingest finishes)
```bash
python scripts/run_experiments.py --embedder st --llm ollama --model qwen3:8b --limit 5     # smoke first
python scripts/run_experiments.py --embedder st --llm ollama --model qwen3:8b               # all of E1..E8
python scripts/run_experiments.py --embedder st --llm ollama --model qwen3:8b --experiments E3,E5   # one at a time
```
Files written under `results/`: `e1_inventory.json`, `e2_chunking.csv`, `e3_retrieval.csv`, `e4_transcript.md`, `e5_answers.csv` + `e5_summary.json`, `e6_benchmark.json`, `e7_ablation.csv`, `e8_iterations.md`. Add `--llm hosted` for the gpt-oss fallback, `--judge` to let the LLM judge answers that do not string-match.
LLM calls take about 0.5 s each (measured in S2). E6 makes 5 calls per golden question, E7 five variants, E8 several rounds: keep `--experiments` small if time is short.

## 4. End of session: copy results out through git
```bash
cd ~/multimodal-rag-group22
git checkout -b results/$(date +%m%d)-$USER
git add results && git status --short           # check: no keys, no tokens, no .env
git -c user.name="<your name>" -c user.email="<your email>" commit -m "results from lab session"
git push -u origin HEAD                          # needs a GitHub token in your own account; do not paste it in chat
```
If the push asks for credentials and you have none, commit only, and say so; the file browser cannot download.
