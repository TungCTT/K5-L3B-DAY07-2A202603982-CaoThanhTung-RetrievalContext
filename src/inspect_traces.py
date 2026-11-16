"""Print human-readable trace details for manual grading of test cases."""
from __future__ import annotations

import json
from pathlib import Path


def dump_traces(label: str, trace_path: Path, cases_path: Path) -> None:
    cases = {
        c["id"]: c
        for c in json.loads(cases_path.read_text(encoding="utf-8"))
        if c["split"] == "test"
    }
    traces = json.loads(trace_path.read_text(encoding="utf-8"))
    print(f"================ {label} ================")
    for t in traces:
        c = cases[t["query_id"]]
        print(
            f"--- {t['query_id']} | answerable={c['answerable']} | "
            f"status={t['status']} | ctx_gold={t.get('context_has_gold')} ---"
        )
        print("Q:", c["question"])
        if c["answerable"]:
            print("EXPECTED:", c["expected_answer"])
            print("GOLD:", [(g["page"], g["evidence"][:140]) for g in c["gold"]])
        else:
            print("ABSENCE_REASON:", c["absence_reason"])
        print("ANSWER:", t["answer"])
        print("CITATIONS:", t["citations"])
        ctx_map = {x["id"]: x for x in t["context"]}
        for cid in t["citations"]:
            ch = ctx_map.get(cid)
            page_str = ch["page"] if ch else "?"
            snippet = ch["text"][:260].replace("\n", " ") if ch else "MISSING"
            print(f"  CITED [{cid}] (p.{page_str}): {snippet}")
        print()


if __name__ == "__main__":
    cases_p = Path("starter/data/converted/agent_cases.json")
    dump_traces("BASELINE", Path("starter/results/agent-test-baseline/traces.json"), cases_p)
    dump_traces("C2", Path("starter/results/agent-test-c2/traces.json"), cases_p)
