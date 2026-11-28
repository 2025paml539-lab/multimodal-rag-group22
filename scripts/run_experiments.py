"""Run the reported experiments. Every number in the report comes from a file this script writes under results/.

Each experiment writes its own file and can be re-run alone, so a lost lab session costs only that experiment:
    E1 e1_inventory.json     E2 e2_chunking.csv      E3 e3_retrieval.csv     E4 e4_transcript.md
    E5 e5_answers.csv        E6 e6_benchmark.json    E7 e7_ablation.csv      E8 e8_iterations.md
    E5B e5_auto_eval.csv + e5_auto_eval_summary.json (LLM-as-judge, never sees the golden answer, compared with E5)
    E9 e9_guardrails.csv

Laptop check (no GPU, no model):  python scripts/run_experiments.py --embedder hash --llm fake --limit 5
Lab (after scripts/ingest_corpus.py):  python scripts/run_experiments.py --embedder st --llm ollama --model qwen3:8b
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from rag import (ChunkConfig, ConversationMemory, FakeLLMClient, Hit, HashEmbedder, HybridIndex,  # noqa: E402
                 Modality, OllamaClient, chunk_elements)
from rag.chunking import count_tokens  # noqa: E402
from rag.eval_metrics import (cites_retrieved, correctness, faithfulness, is_refusal, load_golden, mrr,  # noqa: E402
                              recall_at_k, summarise)
from rag.index import _element_from_dict  # noqa: E402
from rag.pipeline import answer_question  # noqa: E402
from rag.auto_eval import compare_with_golden, judge  # noqa: E402
from rag.prompts import ABLATION_STYLES, VARIANTS, load_ontology, render_prompt  # noqa: E402

CONFIGS = {"A": ChunkConfig(200, 40), "B": ChunkConfig(60, 10)}
ALL = ["E1", "E2", "E3", "E4", "E5", "E5B", "E6", "E7", "E8", "E9"]
_BLOCK = re.compile(r"\[([^\[\],]+), p\.(\d+)\] \(\w+\)\n(.+)")


def fake_llm() -> FakeLLMClient:
    """Extractive stand-in: quotes the first context block with its citation. Only for laptop checks of the plumbing."""
    def respond(prompt: str) -> str:
        if "Rewrite the last question" in prompt:
            return prompt.rsplit("Last question:", 1)[-1].split("\n")[0].strip()
        if "ONE short general rule" in prompt:
            return "Quote numbers exactly as written in the context."
        m = _BLOCK.search(prompt[prompt.rfind("Context:"):])  # context after the last marker, not a few-shot example
        if not m:
            return "Not found in the documents."
        doc, page, text = m.group(1), int(m.group(2)), " ".join(m.group(3).split()[:12])
        if prompt.rstrip().endswith("JSON:"):
            return json.dumps({"answer": text, "citations": [{"document": doc, "page": page}], "refusal": False})
        return f"{text} [{doc}, p.{page}]"
    return FakeLLMClient(respond)


def make_llm(a):
    if a.llm == "fake":
        return fake_llm()
    if a.llm == "ollama":
        return OllamaClient(a.model)
    from rag import HostedOpenAICompatibleClient
    return HostedOpenAICompatibleClient()


def make_embedder(a):
    if a.embedder == "st":
        from rag.embedding import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(a.st_model)
    return HashEmbedder()


def write_csv(path: Path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}", flush=True)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {path}", flush=True)


def pct(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))] if v else 0.0


class Ctx:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.results)
        self.embedder = make_embedder(a)
        self.llm = make_llm(a)
        self.ontology = load_ontology(str(ROOT / "ontology.yaml")) if (ROOT / "ontology.yaml").exists() else None
        parsed = sorted(Path(a.cache, "parsed").glob("*.json"))
        if not parsed:
            raise SystemExit(f"no parsed documents in {a.cache}/parsed. Run scripts/ingest_corpus.py first.")
        self.elements = [_element_from_dict(d) for f in parsed for d in json.loads(f.read_text(encoding="utf-8"))]
        self.golden = load_golden(a.golden)
        if a.limit:
            self.golden = self.golden[:a.limit]
        self._indexes = {}

    def index(self, config: str = "A") -> HybridIndex:
        if config not in self._indexes:
            saved = Path(self.a.index_path)
            if config == "A" and (saved / "elements.json").exists():
                self._indexes[config] = HybridIndex.load(str(saved), self.embedder)
            else:
                idx = HybridIndex(self.embedder)
                idx.add(chunk_elements(self.elements, CONFIGS[config]))
                self._indexes[config] = idx
        return self._indexes[config]

    def judge(self):
        return self.llm if self.a.judge else None


def retrieval_row(hits, g, k):
    return {"recall": recall_at_k(hits, g["doc_id"], g["page"], k), "mrr": mrr(hits, g["doc_id"], g["page"])}


def eval_golden(ctx, golden, variant, mode="hybrid", modalities=None, extra_rules="", index=None):
    rows = []
    for g in golden:
        r = answer_question(g["question"], index or ctx.index(), ctx.llm, variant, ctx.a.k, mode, modalities,
                            ontology=ctx.ontology, extra_rules=extra_rules)
        rows.append({"id": g["id"], "split": g["split"], "modality": g["modality"], **retrieval_row(r["hits"], g, ctx.a.k),
                     "correct": correctness(r["answer"], g["answer"], ctx.judge()),
                     "faithful": faithfulness(r["answer"], r["hits"]),
                     "cited_ok": float(cites_retrieved(r["answer"], r["hits"])),
                     "refused": float(is_refusal(r["answer"])), "total_s": round(r["total_s"], 3),
                     "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
                     "answer": r["answer"].replace("\n", " ")[:300]})
    return rows


def e1(ctx):
    docs = {}
    for e in ctx.elements:
        d = docs.setdefault(e.doc_id, {"pages": 0, "text": 0, "table": 0, "image": 0})
        d["pages"] = max(d["pages"], e.page)
        d[e.modality.value] += 1
    meta = {}
    mp = ROOT / "corpus" / "doc_metadata.csv"
    if mp.exists():
        meta = {r["doc_id"]: r for r in csv.DictReader(open(mp, encoding="utf-8"))}
    by_type = {}
    for doc_id, d in docs.items():
        t = meta.get(doc_id, {}).get("doc_type", "unknown")
        b = by_type.setdefault(t, {"documents": 0, "pages": 0, "text": 0, "table": 0, "image": 0})
        b["documents"] += 1
        for key in ("pages", "text", "table", "image"):
            b[key] += d[key]
    chunks = chunk_elements(ctx.elements, CONFIGS["A"])
    write_json(ctx.out / "e1_inventory.json", {
        "documents": len(docs), "pages": sum(d["pages"] for d in docs.values()),
        "elements": {m.value: sum(d[m.value] for d in docs.values()) for m in Modality},
        "chunks_config_A": {m.value: sum(1 for c in chunks if c.modality == m) for m in Modality},
        "by_doc_type": by_type})


def e2(ctx):
    rows = []
    for name, cfg in CONFIGS.items():
        idx = ctx.index(name)
        chunks = idx.bm25.elements
        rec = [retrieval_row(idx.search(g["question"], k=ctx.a.k), g, ctx.a.k) for g in ctx.golden]
        rows.append({"config": name, "max_words": cfg.max_tokens, "overlap": cfg.overlap, "chunks": len(chunks),
                     "avg_words": round(statistics.mean(count_tokens(c.text) for c in chunks), 1),
                     "recall": round(statistics.mean(r["recall"] for r in rec), 4),
                     "mrr": round(statistics.mean(r["mrr"] for r in rec), 4), "questions": len(rec)})
    write_csv(ctx.out / "e2_chunking.csv", rows)


def e3(ctx):
    setups = {"bm25": ("bm25", None), "dense": ("dense", None), "hybrid": ("hybrid", None),
              "hybrid_text_only": ("hybrid", [Modality.TEXT])}
    idx, rows = ctx.index(), []
    for name, (mode, mods) in setups.items():
        per = [{**retrieval_row(idx.search(g["question"], k=ctx.a.k, mode=mode, modalities=mods), g, ctx.a.k),
                "modality": g["modality"], "split": g["split"]} for g in ctx.golden]
        summ = summarise(per, keys=("recall", "mrr"))
        for group, vals in summ.items():
            if group != "n":
                rows.append({"setup": name, "group": group, "n": vals["n"], "recall": vals.get("recall"), "mrr": vals.get("mrr")})
    write_csv(ctx.out / "e3_retrieval.csv", rows)


def e4(ctx):
    mem, idx = ConversationMemory(), ctx.index()
    lines = ["# E4 memory follow-up transcript", "", f"LLM: {ctx.a.llm} {ctx.a.model if ctx.a.llm != 'fake' else ''}", ""]
    first = ctx.golden[0]
    turns = [first["question"], "And what about the next page of that document?"]
    for q in turns:
        standalone = mem.standalone_question(q, ctx.llm)
        r = answer_question(standalone, idx, ctx.llm, ctx.a.variant, ctx.a.k, ontology=ctx.ontology)
        lines += [f"**User:** {q}", f"**Rewritten:** {standalone}",
                  f"**Top hits:** {' | '.join(h.citation + ' ' + h.element.modality.value for h in r['hits'][:3])}",
                  f"**Answer:** {r['answer']}", ""]
        mem.add(q, r["answer"])
    (ctx.out / "e4_transcript.md").parent.mkdir(parents=True, exist_ok=True)
    (ctx.out / "e4_transcript.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {ctx.out / 'e4_transcript.md'}", flush=True)


def e5(ctx):
    rows = eval_golden(ctx, ctx.golden, ctx.a.variant)
    write_csv(ctx.out / "e5_answers.csv", rows)
    write_json(ctx.out / "e5_summary.json", {"variant": ctx.a.variant, **summarise(rows, ("recall", "mrr", "correct", "faithful", "cited_ok"))})


def e6(ctx):
    idx, lat, same, ours, naive = ctx.index(), [], [], [], []
    page_text = {}
    for e in ctx.elements:
        page_text.setdefault((e.doc_id, e.page), []).append(e)
    for g in ctx.golden:
        runs = [answer_question(g["question"], idx, ctx.llm, ctx.a.variant, ctx.a.k, ontology=ctx.ontology) for _ in range(5)]
        lat += [r["total_s"] for r in runs]
        same.append(float(len({r["answer"] for r in runs}) == 1))
        r0 = runs[0]
        ours.append(r0["prompt_words"])
        pages = {(h.element.doc_id, h.element.page) for h in r0["hits"]}
        whole = [Hit(e, 1.0, i) for i, e in enumerate(x for p in sorted(pages) for x in page_text.get(p, []))]
        naive.append(len(render_prompt(ctx.a.variant, g["question"], whole, ctx.ontology).split()))
    from rag import guardrails
    limiter = guardrails.RateLimiter(max_requests=10**6)
    g_times, plain_times = [], []
    for g in ctx.golden:
        r = answer_question(g["question"], idx, ctx.llm, ctx.a.variant, ctx.a.k, ontology=ctx.ontology)
        t0 = time.perf_counter()
        guardrails.check_input(g["question"])
        limiter.allow("bench")
        guardrails.check_output(r["answer"], r["hits"], g["question"])
        g_times.append(time.perf_counter() - t0)
        plain_times.append(r["total_s"])
    guard = {"queries": len(g_times), "guard_mean_s": round(statistics.mean(g_times), 6) if g_times else None,
             "guard_p95_s": round(pct(g_times, 0.95), 6), "query_without_guard_p95_s": round(pct(plain_times, 0.95), 3),
             "limit_s": 6.0}
    write_json(ctx.out / "e6_benchmark.json", {
        "queries": len(ctx.golden), "runs_per_query": 5, "llm": ctx.a.llm,
        "latency_p50_s": round(pct(lat, 0.5), 3), "latency_p95_s": round(pct(lat, 0.95), 3), "latency_limit_s": 90,
        "consistency_identical_answers": round(statistics.mean(same), 4) if same else None,
        "prompt_words_ours_mean": round(statistics.mean(ours), 1) if ours else None,
        "prompt_words_naive_whole_pages_mean": round(statistics.mean(naive), 1) if naive else None,
        "token_reduction_pct": round(100 * (1 - sum(ours) / sum(naive)), 1) if naive and sum(naive) else None,
        "guardrails": guard})


def e5b(ctx):
    """Automatic evaluation. The judge never sees the golden answer; its verdict is compared with golden correctness."""
    rows = []
    for g in ctx.golden:
        r = answer_question(g["question"], ctx.index(), ctx.llm, ctx.a.variant, ctx.a.k, ontology=ctx.ontology)
        v = judge(ctx.llm, g["question"], r["answer"], r["hits"])
        ok = None if v is None else int(v["supported"] and v["answers_question"])
        rows.append({"id": g["id"], "split": g["split"], "modality": g["modality"],
                     "correct": correctness(r["answer"], g["answer"], None), "judge_ok": ok,
                     "supported": None if v is None else int(v["supported"]),
                     "answers_question": None if v is None else int(v["answers_question"]),
                     "answer": r["answer"].replace("\n", " ")[:200]})
    write_csv(ctx.out / "e5_auto_eval.csv", rows)
    write_json(ctx.out / "e5_auto_eval_summary.json", compare_with_golden(rows))


def e7(ctx):
    """The five prompt styles named in the brief, same model and questions: quality, tokens, latency, consistency."""
    rows = []
    for style in ABLATION_STYLES:
        r = eval_golden(ctx, ctx.golden, style)
        s = summarise(r, ("correct", "faithful", "cited_ok", "refused"))["all"]
        lat = [x["total_s"] for x in r]
        same = []
        for g in ctx.golden:
            answers = {answer_question(g["question"], ctx.index(), ctx.llm, style, ctx.a.k,
                                       ontology=ctx.ontology)["answer"] for _ in range(5)}
            same.append(float(len(answers) == 1))
        rows.append({"style": style, "n": s["n"], "correct": s["correct"], "faithful": s["faithful"],
                     "cited_ok": s["cited_ok"], "refused": s["refused"],
                     "input_tokens_mean": round(statistics.mean(x["input_tokens"] for x in r), 1) if r else None,
                     "output_tokens_mean": round(statistics.mean(x["output_tokens"] for x in r), 1) if r else None,
                     "latency_p50_s": round(pct(lat, 0.5), 3), "latency_p95_s": round(pct(lat, 0.95), 3),
                     "consistency": round(statistics.mean(same), 4) if same else None})
    write_csv(ctx.out / "e7_ablation.csv", rows)


def e9(ctx):
    from rag import guardrails
    path = ROOT / "eval" / "guardrail_tests.jsonl"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        v = guardrails.check_input(c["question"])
        rows.append({"expected": c["expected"], "got": v.category, "pass": int(v.category == c["expected"]),
                     "question": c["question"].strip()[:80], "reason": v.reason})
    write_csv(ctx.out / "e9_guardrails.csv", rows)
    print(f"  {sum(r['pass'] for r in rows)}/{len(rows)} guardrail cases passed", flush=True)


def e8(ctx):
    dev = [g for g in ctx.golden if g["split"] == "dev"]
    test = [g for g in ctx.golden if g["split"] == "test"]
    if not dev or not test:
        print("  E8 skipped: needs both dev and test questions", flush=True)
        return
    score = lambda rows: statistics.mean(r["correct"] for r in rows) if rows else 0.0
    variant, rules, log = ctx.a.variant, [], []
    best = score(eval_golden(ctx, dev, variant))
    log.append(("0 (baseline)", "", best))
    tried = []   # every rule proposed so far, kept or not, so the model is asked for a different one each round
    for it in range(1, ctx.a.iterations + 1):
        fails = [r for r in eval_golden(ctx, dev, variant, extra_rules="\n".join(rules)) if not r["correct"]]
        if not fails:
            break
        gold = {g["id"]: g["answer"] for g in dev}
        listing = "\n".join(f"- id {r['id']}: gold '{gold[r['id']]}', model said '{r['answer'][:80]}'" for r in fails[:5])
        rule = ctx.llm.generate("A question-answering prompt gave these wrong answers on a development set:\n" + listing +
                                "\nWrite ONE short general rule (one sentence, no examples from the list) that would help."
                                + ("\nRules already tried, do not repeat or paraphrase them:\n" + "\n".join(f"- {t}" for t in tried)
                                   if tried else "") + "\nRule:",
                                max_tokens=60).text.strip().splitlines()[0][:200]
        tried.append(rule)
        trial = score(eval_golden(ctx, dev, variant, extra_rules="\n".join(rules + [rule])))
        kept = trial > best
        if kept:
            rules.append(rule)
            best = trial
        log.append((f"{it} ({'kept' if kept else 'rejected'})", rule, trial))
    base_test = score(eval_golden(ctx, test, variant))
    final_test = score(eval_golden(ctx, test, variant, extra_rules="\n".join(rules)))
    lines = ["# E8 self-improving loop", "", "Rules are chosen on the dev split only. The test split is scored twice, at the end.", "",
             "| iteration | rule proposed | dev correctness |", "|---|---|---|"]
    lines += [f"| {n} | {r or '-'} | {s:.3f} |" for n, r, s in log]
    lines += ["", f"Test correctness without rules: {base_test:.3f}", f"Test correctness with kept rules ({len(rules)}): {final_test:.3f}"]
    (ctx.out / "e8_iterations.md").parent.mkdir(parents=True, exist_ok=True)
    (ctx.out / "e8_iterations.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {ctx.out / 'e8_iterations.md'}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-path", default=os.path.expanduser("~/rag_index"))
    ap.add_argument("--cache", default=os.path.expanduser("~/rag_cache"))
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--golden", default=str(ROOT / "eval"))
    ap.add_argument("--embedder", choices=["hash", "st"], default="hash")
    ap.add_argument("--st-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--llm", choices=["fake", "ollama", "hosted"], default="fake")
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--variant", default="v2_citation_rules", choices=VARIANTS)
    ap.add_argument("--experiments", default="all", help="e.g. E1,E3 (default all)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N golden questions (smoke run)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--judge", action="store_true", help="use the LLM as a correctness judge when string match fails")
    a = ap.parse_args()
    wanted = ALL if a.experiments == "all" else [e.strip().upper() for e in a.experiments.split(",")]
    bad = [e for e in wanted if e not in ALL]
    if bad:
        raise SystemExit(f"unknown experiments {bad}; choose from {ALL}")
    ctx = Ctx(a)
    print(f"{len(ctx.elements)} elements, {len(ctx.golden)} golden questions, llm={a.llm}, embedder={a.embedder}", flush=True)
    fns = {"E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5, "E5B": e5b, "E6": e6, "E7": e7, "E8": e8, "E9": e9}
    for name in wanted:
        t = time.perf_counter()
        print(f"{name} ...", flush=True)
        fns[name](ctx)
        print(f"{name} done in {time.perf_counter() - t:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
