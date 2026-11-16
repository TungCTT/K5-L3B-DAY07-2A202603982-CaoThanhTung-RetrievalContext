"""Benchmark retrieval methods (bm25, dense, hybrid) on dev/test splits."""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from retrievers import BM25Retriever, DensePgVectorRetriever, HybridRRFRetriever

DEFAULT_DB_URL = "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab"
WS_RE = re.compile(r"\s+")


def norm_text(text: str) -> str:
    return WS_RE.sub(" ", text).strip()


def has_evidence(chunk: dict[str, Any], gold_list: list[dict[str, Any]]) -> bool:
    c_text = norm_text(chunk["text"])
    for g in gold_list:
        if (
            chunk["source_id"] == g["source_id"]
            and chunk["page"] == g["page"]
            and norm_text(g["evidence"]) in c_text
        ):
            return True
    return False


def build_context(ranked: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    used = 0
    for ch in ranked:
        if ch["id"] in seen:
            continue
        formatted = f"[{ch['id']} | {ch['source']} p.{ch['page']}] {ch['text']}"
        if used + len(formatted) > budget:
            continue
        seen.add(ch["id"])
        selected.append(ch)
        used += len(formatted)
    return selected


def evaluate_retriever(
    name: str,
    retriever: Any,
    questions: list[dict[str, Any]],
    top_k: int = 3,
    context_budget: int = 2400,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hit1 = 0
    hit3 = 0
    context_hit = 0
    latencies_ms: list[float] = []
    context_chars: list[int] = []
    traces: list[dict[str, Any]] = []

    t_all = time.perf_counter()
    for q in questions:
        t0 = time.perf_counter()
        results = retriever.search(q["question"], top_k=top_k, filters=q.get("filter"))
        ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(ms)

        gold = q["gold"]
        h1 = bool(results) and has_evidence(results[0], gold)
        h3 = any(has_evidence(c, gold) for c in results[:3])
        ctx = build_context(results, budget=context_budget)
        ch = any(has_evidence(c, gold) for c in ctx)
        ctx_len = sum(len(f"[{c['id']} | {c['source']} p.{c['page']}] {c['text']}") for c in ctx)
        context_chars.append(ctx_len)

        hit1 += int(h1)
        hit3 += int(h3)
        context_hit += int(ch)
        traces.append(
            {
                "id": q["id"],
                "question": q["question"],
                "filter": q.get("filter"),
                "hit1": h1,
                "hit3": h3,
                "context_hit": ch,
                "context_chars": ctx_len,
                "latency_ms": round(ms, 2),
                "top_k": [
                    {
                        "rank": c["rank"],
                        "id": c["id"],
                        "page": c["page"],
                        "score": c["score"],
                        "metadata": c.get("metadata", {}),
                    }
                    for c in results
                ],
            }
        )

    total_s = time.perf_counter() - t_all
    n = len(questions)
    summary = {
        "method": name,
        "n": n,
        "top_k": top_k,
        "context_budget": context_budget,
        "hit1": hit1,
        "hit1_rate": round(hit1 / n, 4) if n else 0.0,
        "hit3": hit3,
        "hit3_rate": round(hit3 / n, 4) if n else 0.0,
        "context_hit": context_hit,
        "context_hit_rate": round(context_hit / n, 4) if n else 0.0,
        "avg_context_chars": round(sum(context_chars) / n, 1) if n else 0.0,
        "median_ms": round(statistics.median(latencies_ms), 2) if latencies_ms else 0.0,
        "total_seconds": round(total_s, 3),
        "service_cost_usd": 0.0,
    }
    return summary, traces


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate BM25, Dense, and Hybrid retrieval.")
    parser.add_argument("--chunks", type=Path, default=Path("starter/data/converted/chunks.json"))
    parser.add_argument("--questions", type=Path, default=Path("starter/data/converted/questions.json"))
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=2400)
    parser.add_argument("--output", type=Path, default=Path("starter/results/retrieval-dev"))
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    all_q = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = [q for q in all_q if q["split"] == args.split]

    db_url = os.getenv("DATABASE_URL", DEFAULT_DB_URL)
    emb_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    bm25 = BM25Retriever(chunks)
    dense = DensePgVectorRetriever(database_url=db_url, model_name=emb_model, strategy="C")
    hybrid = HybridRRFRetriever(dense=dense, lexical=bm25, candidate_k=20, rrf_k=60)

    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, ret in (("bm25", bm25), ("dense", dense), ("hybrid_rrf", hybrid)):
        summary, traces = evaluate_retriever(
            name=name,
            retriever=ret,
            questions=questions,
            top_k=args.top_k,
            context_budget=args.budget,
        )
        summaries.append(summary)
        out_dir = args.output / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (args.output / "comparison.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
