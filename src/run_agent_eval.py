"""Run Agentic RAG evaluation on dev or test split and save traces."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

from agent import run_agent_case
from eval_retrieval import has_evidence
from llm_client import DeepSeekClient
from retrievers import BM25Retriever, DensePgVectorRetriever, HybridRRFRetriever

DEFAULT_DB_URL = "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab"


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run Agentic RAG on benchmark split.")
    parser.add_argument("--chunks", type=Path, default=Path("starter/data/converted/chunks.json"))
    parser.add_argument("--cases", type=Path, default=Path("starter/data/converted/agent_cases.json"))
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--retriever", choices=("bm25", "dense", "hybrid_rrf"), default="dense")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--budget", type=int, default=2400)
    parser.add_argument("--context-mode", choices=("fifo", "ranked"), default="ranked")
    parser.add_argument("--output", type=Path, default=Path("starter/results/agent-dev-dense"))
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    all_cases = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = [c for c in all_cases if c["split"] == args.split]

    db_url = os.getenv("DATABASE_URL", DEFAULT_DB_URL)
    emb_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    if args.retriever == "bm25":
        retriever = BM25Retriever(chunks)
    elif args.retriever == "dense":
        retriever = DensePgVectorRetriever(database_url=db_url, model_name=emb_model, strategy="C")
    else:
        bm25 = BM25Retriever(chunks)
        dense = DensePgVectorRetriever(database_url=db_url, model_name=emb_model, strategy="C")
        retriever = HybridRRFRetriever(dense=dense, lexical=bm25, candidate_k=20, rrf_k=60)

    llm = DeepSeekClient()
    traces = []
    t0 = time.perf_counter()

    for case in cases:
        tr = run_agent_case(
            case=case,
            retriever=retriever,
            llm=llm,
            top_k=args.top_k,
            max_searches=args.max_searches,
            context_budget=args.budget,
            context_mode=args.context_mode,
        )
        # Annotate whether final context contains gold evidence (for answerable cases)
        if case.get("answerable", True) and case.get("gold"):
            tr["context_has_gold"] = any(has_evidence(c, case["gold"]) for c in tr["context"])
        else:
            tr["context_has_gold"] = None
        traces.append(tr)
        print(
            f"[{tr['query_id']}] status={tr['status']} searches={len(tr['events'])} "
            f"cites={len(tr['citations'])} ctx_gold={tr['context_has_gold']} "
            f"tokens={tr['total_tokens']} ms={tr['latency_ms']}",
            flush=True,
        )

    total_s = time.perf_counter() - t0
    ans_traces = [t for t, c in zip(traces, cases) if c.get("answerable", True)]
    abs_traces = [t for t, c in zip(traces, cases) if not c.get("answerable", True)]
    errors = [t for t in traces if t["status"] == "error"]

    summary = {
        "split": args.split,
        "model": llm.model,
        "retriever": args.retriever,
        "top_k": args.top_k,
        "max_searches": args.max_searches,
        "context_budget": args.budget,
        "context_mode": args.context_mode,
        "total_cases": len(cases),
        "errors": len(errors),
        "tool_use_rate": round(
            sum(1 for t in traces if t["status"] != "error" and any(e["type"] == "search" for e in t["events"]))
            / max(len(traces) - len(errors), 1),
            4,
        ),
        "answerable_context_hit": sum(1 for t in ans_traces if t["context_has_gold"]),
        "answerable_total": len(ans_traces),
        "answerable_answered": sum(1 for t in ans_traces if t["status"] == "answered"),
        "answerable_false_abstain": sum(1 for t in ans_traces if t["status"] == "abstain"),
        "unanswerable_abstained": sum(1 for t in abs_traces if t["status"] == "abstain"),
        "unanswerable_total": len(abs_traces),
        "total_tokens": sum(t["total_tokens"] for t in traces),
        "median_latency_ms": round(statistics.median(t["latency_ms"] for t in traces), 2) if traces else 0.0,
        "total_seconds": round(total_s, 2),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nSUMMARY:", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
