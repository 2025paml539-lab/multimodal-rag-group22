"""Streamlit demo and project pages. Run from the repository root:

    pip install streamlit pandas numpy
    streamlit run app/streamlit_app.py

Needs a downloaded index folder (elements.json, dense.npy) from the lab: ~/rag_index. See docs/DEMO.md.
Everything shown comes from files in this repository (results/, eval/, corpus/, docs/) or from the live pipeline.
"""
import csv
import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from rag.demo import SAMPLES, STYLES, build_system, recorded_answers, to_markdown  # noqa: E402

st.set_page_config(page_title="Group 22: banking document QA", layout="wide")
FINAL = ROOT / "results" / "final_30"


def read_csv(path):
    return pd.read_csv(path) if Path(path).exists() else None


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None


@st.cache_resource(show_spinner="Loading the index ...")
def get_system(index_path, embedder, llm, model, host):
    return build_system(index_path, embedder, llm, model, host)


page = st.sidebar.radio("Page", ["Ask (live demo)", "Results", "Corpus and questions", "How we built it"])
st.sidebar.markdown("---")

# ---------------------------------------------------------------- Ask
if page == "Ask (live demo)":
    st.title("Ask the bank documents")
    st.caption("Indian banking regulation. Every answer must cite a document and page, or refuse.")
    index_path = st.sidebar.text_input("Index folder", str(ROOT / "demo_index"))
    has_index = (Path(index_path) / "elements.json").exists()
    mode = st.sidebar.selectbox("Answer source", ["Retrieval only (no model)", "Live: Ollama on this machine",
                                                  "Live: hosted open-weight endpoint (env vars)", "Replay answers recorded in the lab"],
                                index=0 if has_index else 3)  # no downloaded index (for example on Streamlit Cloud): open on the replay
    model = st.sidebar.text_input("Ollama model", "qwen3:8b")
    host = st.sidebar.text_input("Ollama host", "http://localhost:11434")
    style = st.sidebar.selectbox("Prompt style", STYLES, index=STYLES.index("create"))
    if mode.startswith("Replay"):
        rows = recorded_answers(ROOT)
        st.info("These answers were produced by qwen3:8b on the lab A100 during the final run (results/final_30/e5_answers.csv). "
                "Nothing is generated now.")
        if not rows:
            st.warning("No recorded results found.")
        else:
            pick = st.selectbox("Golden question", [f"{r['id']}: {r['question']}" for r in rows])
            r = next(x for x in rows if pick.startswith(x["id"] + ":"))
            st.markdown(f"**Recorded answer:** {r['recorded_answer']}")
            st.markdown(f"**Gold answer:** {r['gold']}  |  **source:** {r['source']}  |  **marked correct:** {r['correct']}  |  "
                        f"**gold page retrieved in top 5:** {r['recall'] == '1.0'}")
    elif not (Path(index_path) / "elements.json").exists():
        st.warning(f"No index at {index_path}. Zip ~/rag_index in the lab, download it, unzip into {ROOT / 'demo_index'} (see docs/DEMO.md).")
    else:
        llm = {"Retrieval only (no model)": "none", "Live: Ollama on this machine": "ollama"}.get(mode, "hosted")
        try:
            system = get_system(index_path, "auto", llm, model, host)
        except Exception as e:  # missing env vars, no Ollama and similar
            st.error(f"Could not start: {e}")
            st.stop()
        st.sidebar.caption(f"Retrieval: {system.llm_label} | {'hybrid' if system.mode == 'hybrid' else 'BM25 only'} | {len(system.index)} chunks")
        c1, c2 = st.columns([3, 1])
        sample = c1.selectbox("Example questions (or type your own below)", [""] + SAMPLES)
        q = c1.text_input("Question", value=sample)
        if c2.button("Forget the conversation"):
            system.reset()
        if st.button("Ask") and q.strip():
            with st.spinner("Working ..."):
                r = system.ask(q, variant=style)
            st.markdown(to_markdown(r))

# ---------------------------------------------------------------- Results
elif page == "Results":
    st.title("Results from the real lab runs")
    st.caption("A100 80 GB, qwen3:8b through Ollama, MiniLM embeddings. Final run: 30 golden questions (results/final_30). Files, not estimates.")
    e5 = read_json(FINAL / "e5_summary.json")
    e6 = read_json(FINAL / "e6_benchmark.json")
    if e5 and e6:
        a = e5["all"]
        c = st.columns(5)
        c[0].metric("Recall@5", a["recall"]); c[1].metric("Answers correct", a["correct"])
        c[2].metric("P95 latency (s)", e6["latency_p95_s"]); c[3].metric("Identical answers over 5 runs", e6["consistency_identical_answers"])
        c[4].metric("Fewer words than whole pages", f"{e6['token_reduction_pct']}%")
    st.subheader("Prompt styles (E7)")
    e7 = read_csv(FINAL / "e7_ablation.csv")
    if e7 is not None:
        st.dataframe(e7, hide_index=True)
        st.bar_chart(e7.set_index("style")[["correct", "faithful", "cited_ok", "consistency"]])
    st.subheader("Retrieval methods (E3)")
    e3 = read_csv(FINAL / "e3_retrieval.csv")
    if e3 is not None:
        st.dataframe(e3[e3["group"].isin(["all", "modality:text", "modality:table", "modality:image"])], hide_index=True)
    st.subheader("Chunking (E2)")
    e2 = read_csv(FINAL / "e2_chunking.csv")
    if e2 is not None:
        st.dataframe(e2, hide_index=True)
    st.subheader("Memory demonstration (E4)")
    p = FINAL / "e4_transcript.md"
    if p.exists():
        st.markdown(p.read_text(encoding="utf-8"))
    st.subheader("Self-improving loop (E8)")
    p = FINAL / "e8_iterations.md"
    if p.exists():
        st.markdown(p.read_text(encoding="utf-8"))
    st.subheader("Guardrails (E9, 21 cases we wrote)")
    e9 = read_csv(ROOT / "results" / "e9_guardrails.csv")
    if e9 is not None:
        st.write(f"{int(e9['pass'].sum())} of {len(e9)} cases passed.")
        st.dataframe(e9[["expected", "got", "question"]], hide_index=True)
    st.subheader("Integrated demonstration (E10)")
    p = ROOT / "results" / "e10" / "e10_demo.md"
    if p.exists():
        st.markdown(p.read_text(encoding="utf-8"))
    st.caption("Limits: 30 questions of which 2 are image-based; one model; answers identical in 90 percent of repeated runs.")

# ---------------------------------------------------------------- Corpus
elif page == "Corpus and questions":
    st.title("Corpus and golden questions")
    inv = read_json(ROOT / "results" / "e1_inventory.json")
    if inv:
        c = st.columns(4)
        c[0].metric("Documents parsed", inv["documents"]); c[1].metric("Pages", inv["pages"])
        c[2].metric("Table elements", inv["elements"]["table"]); c[3].metric("Image elements", inv["elements"]["image"])
        st.write("By document type:", pd.DataFrame(inv["by_doc_type"]).T)
    st.subheader("Document list")
    man = read_csv(ROOT / "corpus" / "manifest.csv")
    meta = read_csv(ROOT / "corpus" / "doc_metadata.csv")
    if man is not None and meta is not None:
        df = man.merge(meta[["doc_id", "doc_type", "topic", "sensitivity"]], on="doc_id", how="left")
        kind = st.multiselect("Document type", sorted(df["doc_type"].dropna().unique()))
        if kind:
            df = df[df["doc_type"].isin(kind)]
        st.dataframe(df[["doc_id", "title", "type", "pages", "doc_type", "topic", "sensitivity"]], hide_index=True)
        st.caption("sensitivity = synthetic marks the 20 decks we generated. 18 extra RBI documents were added without instructor approval.")
    st.subheader("Golden set (30 questions, each opened in the PDF page)")
    from rag.eval_metrics import load_golden
    g = pd.DataFrame(load_golden(str(ROOT / "eval")))
    st.dataframe(g[["id", "modality", "split", "question", "answer", "doc_id", "page"]], hide_index=True)

# ---------------------------------------------------------------- About
else:
    st.title("How we built it")
    st.markdown("""
**Problem.** A compliance or customer-support analyst needs a sourced answer from long Indian banking documents (RBI Master Directions,
annual reports, booklets, decks). Answers must cite document and page, or refuse. The instructor approved the corpus and scenario on 21 Nov 2026.

**Approach.** Parse text, tables and images; OCR for text inside images; chunk; embed with MiniLM; index with BM25 and dense vectors fused by rank;
hybrid retrieval with metadata; a prompt filled with the retrieved chunks; a local open-weight model; input and output guardrails and a rate limiter.
Retrieval and not fine-tuning, because the sources are long, change, and every answer must be traceable to a page.
""")
    st.subheader("Timeline (from git history)")
    st.table(pd.DataFrame([
        ["21 Nov", "Repository, plan, lab guide; instructor approval received"],
        ["22 Nov", "Core code: elements, chunking, parsers, hybrid retrieval, memory, LLM client; offline tests"],
        ["23 Nov", "Lab session S2: environment check, parsers on real files, table-text bug found and fixed; 130 RBI documents catalogued"],
        ["24 to 27 Nov", "Ontology, metadata, 8 golden questions, extra documents and synthetic decks"],
        ["28 Nov", "Lab session: speed problems found and fixed, full ingest, experiments E1 to E10, 30 golden questions, report draft, notebook"],
    ], columns=["Date", "What"]))
    st.subheader("Resources")
    st.markdown("""
| | |
|---|---|
| Compute | BITS lab notebook: NVIDIA A100-SXM4-80GB, driver 535.309.01, 14 GB memory, 30 GB disk, 7.2 CPUs allowed (the lab reports 256) |
| Lab sessions | 23 Nov (S2) and 28 Nov (session started 09:01 UTC by the process clock; experiments ran until about 13:30 UTC). **Exact booked hours: fill from the lab booking portal.** |
| Software | PyTorch 2.8.0+cu128, Ollama 0.12.6, qwen3:8b (5.2 GB), sentence-transformers all-MiniLM-L6-v2, PyMuPDF, python-pptx, RapidOCR, Jinja2 (full list: results/lab_packages.txt) |
| Corpus | 166 of 167 files parsed, 8,307 pages, 20,339 chunks. 20 decks are synthetic. OCR text exists only for the decks and the first documents |
| Code | 90+ offline tests (fake model, they check the code and are not results); all reported numbers come from real lab runs in results/ |
""")
    st.subheader("What went wrong and what we changed")
    st.markdown("""
- The first answer check marked correct answers wrong ("three" against "3", the rupee sign against "Rs"). We normalised them and kept the strict result (0.375 against 0.875 on the first 8 questions).
- A table-reading fix rescanned the page for every cell and took 102 of 112 seconds on one PDF. We read the words once per page.
- The lab shows 256 CPUs but allows 7; libraries started too many threads. Pinning to 6 CPUs took one PDF from 68 s to 4.8 s.
- Table and chart slides did not repeat the deck name, so table recall on the synthetic decks was 0.0. Indexing each chunk with its document title gave 1.0.
- A keyword scope check refused 5 of 30 real questions. Scope is now decided together with retrieval.
- Not solved: answers identical in only 90 percent of repeated runs, OCR for most PDFs, D147 does not parse, only 2 image-based golden questions, the off-topic "biryani" question reached the model before being refused.
""")
    st.subheader("Use of an AI coding assistant")
    st.markdown("An AI coding assistant (Kiro) wrote most of the code and the first drafts of the documents, under our direction. "
                "We ran every experiment in the lab, checked each golden answer against its PDF page, and wrote the conclusions. "
                "Each member's own contribution is stated in the signed contribution statement.")
    st.subheader("Team")
    st.markdown("Nandhaluru Kishore (2025paml539): core pipeline and lab. Vishruth M (2025paml536): corpus, ontology, data. "
                "Satya Prakash (2025paml561): evaluation and prompts. Takkars Bhavani (2025paml562): guardrails, design, report. "
                "Planned roles; the confirmed split is in the contribution statement.")
