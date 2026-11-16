"""Normalize event='search' on existing trace files for starter/agent_metrics.py."""
from __future__ import annotations

import json
from pathlib import Path

for p_str in (
    "starter/results/agent-test-baseline/traces.json",
    "starter/results/agent-test-c2/traces.json",
):
    p = Path(p_str)
    data = json.loads(p.read_text(encoding="utf-8"))
    for t in data:
        for e in t.get("events", []):
            if "event" not in e and e.get("type") == "search":
                e["event"] = "search"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("Normalized events.")
