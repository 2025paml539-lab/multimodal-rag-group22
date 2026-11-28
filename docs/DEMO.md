# Running the demo

## A. In the lab (real GPU, real model): notebooks/live_demo.ipynb
1. Book a lab session, open JupyterLab, open a terminal: `source ~/venv_rag/bin/activate; cd ~/multimodal-rag-group22; git pull; bash scripts/lab_start.sh qwen3:8b`.
2. Open `notebooks/live_demo.ipynb`, run the cells, use the form. Share your screen for the video.
Streamlit cannot be opened from the lab: `.../proxy/8501/` returned 404 in session S2 (no jupyter-server-proxy).

## B. On a laptop (any time): app/streamlit_app.py
1. In the lab, zip the index: `cd ~ && zip -qr rag_index.zip rag_index`, download it from the file browser, unzip into `demo_index/` in the repository root (it holds `elements.json` and `dense.npy`).
2. `pip install streamlit pandas numpy jinja2 pyyaml` (add `sentence-transformers` for dense retrieval; without it the app uses BM25 only and says so).
3. `streamlit run app/streamlit_app.py`.
4. Answer source: *Replay* needs nothing else and shows answers recorded in the lab (labelled). *Retrieval only* shows the evidence and guardrail verdict. *Live: Ollama* needs Ollama and qwen3:8b on the laptop. *Live: hosted* needs `RAG_LLM_BASE_URL`, `RAG_LLM_MODEL`, `RAG_LLM_API_KEY` in the environment (the key is never stored).

The pages Results, Corpus and How we built it read the files in `results/`, `corpus/`, `eval/`; they need no index.
