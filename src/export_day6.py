"""Export chunks from Day 6 PostgreSQL (pgvector) to Day 7 raw JSON format."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

DEFAULT_DB_URL = "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab"


def export_chunks(database_url: str, strategy: str, output_path: Path) -> int:
    """Query chunks for a given Day 6 strategy and write them as a JSON array."""
    sql = """
    SELECT chunk_id, document_id, source, page, chapter, content_type, text, metadata
    FROM chunks
    WHERE strategy = %s
    ORDER BY page, chunk_id;
    """
    rows_out: list[dict] = []
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (strategy,))
            for chunk_id, document_id, source, page, chapter, content_type, text, metadata in cur.fetchall():
                meta = dict(metadata) if isinstance(metadata, dict) else {}
                if chapter is not None and "chapter" not in meta:
                    meta["chapter"] = chapter
                if content_type is not None and "content_type" not in meta:
                    meta["content_type"] = content_type
                rows_out.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "source": source,
                        "page": page,
                        "text": text,
                        "metadata": meta,
                    }
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(rows_out)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Export Day 6 pgvector chunks to JSON.")
    parser.add_argument("--strategy", default="C", help="Day 6 strategy to export (default: C)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("starter/data/raw_day6/chunks.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", DEFAULT_DB_URL),
        help="PostgreSQL connection string",
    )
    args = parser.parse_args()
    count = export_chunks(args.database_url, args.strategy, args.output)
    print(f"Exported {count} chunks (strategy={args.strategy}) to {args.output}")


if __name__ == "__main__":
    main()
