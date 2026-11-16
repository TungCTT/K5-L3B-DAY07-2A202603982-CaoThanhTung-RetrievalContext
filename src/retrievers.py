"""Retrieval implementations for Day 7: BM25 Lexical, Dense (pgvector), and Hybrid (RRF)."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import psycopg
from sentence_transformers import SentenceTransformer

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
ALLOWED_FILTER_COLUMNS = {"chapter", "content_type", "document_id", "page"}
TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Simple lowercase word tokenizer."""
    return TOKEN_RE.findall(text.lower())


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def _matches_filter(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    meta = chunk.get("metadata") or {}
    for key, val in filters.items():
        if key not in ALLOWED_FILTER_COLUMNS or val is None:
            continue
        if key == "document_id":
            if chunk.get("source_id") != val:
                return False
        elif key == "page":
            if chunk.get("page") != val:
                return False
        else:
            if meta.get(key) != val:
                return False
    return True


class BM25Retriever:
    """In-memory Okapi BM25 lexical retriever with optional metadata filtering."""

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._doc_tokens: list[list[str]] = [tokenize(c["text"]) for c in chunks]
        self._doc_tf: list[Counter[str]] = [Counter(toks) for toks in self._doc_tokens]
        self._doc_len: list[int] = [len(toks) for toks in self._doc_tokens]
        self._avgdl: float = sum(self._doc_len) / max(len(self._doc_len), 1)
        df: Counter[str] = Counter()
        for tf in self._doc_tf:
            df.update(tf.keys())
        n_docs = len(chunks)
        self._idf: dict[str, float] = {
            term: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def search(
        self,
        query: str,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for idx, chunk in enumerate(self.chunks):
            if not _matches_filter(chunk, filters):
                continue
            tf = self._doc_tf[idx]
            dl = self._doc_len[idx]
            score = 0.0
            for term in q_tokens:
                if term not in tf:
                    continue
                idf = self._idf.get(term, 0.0)
                freq = tf[term]
                denom = freq + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
                score += idf * (freq * (self.k1 + 1.0)) / max(denom, 1e-9)
            if score > 0.0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "id": ch["id"],
                "source_id": ch["source_id"],
                "source": ch["source"],
                "page": ch["page"],
                "text": ch["text"],
                "metadata": ch.get("metadata", {}),
                "score": round(score, 6),
                "rank": rank,
            }
            for rank, (score, ch) in enumerate(scored[:top_k], start=1)
        ]


class DensePgVectorRetriever:
    """Dense retriever querying Day 6 PostgreSQL + pgvector with BGE embeddings."""

    def __init__(
        self,
        database_url: str,
        model_name: str = "BAAI/bge-small-en-v1.5",
        strategy: str = "C",
    ) -> None:
        if not database_url:
            raise ValueError("database_url must be provided")
        self.database_url = database_url
        self.model_name = model_name
        self.strategy = strategy
        self._model = SentenceTransformer(model_name)
        self._use_bge_prefix = "bge-" in model_name.lower()

    def embed_query(self, query: str) -> list[float]:
        text = f"{BGE_QUERY_PREFIX}{query}" if self._use_bge_prefix else query
        vec = self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return vec.tolist()

    def search(
        self,
        query: str,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        q_vec = self.embed_query(query)
        q_lit = _vector_literal(q_vec)
        where_clauses = ["strategy = %s"]
        params: list[Any] = [q_lit, self.strategy]

        if filters:
            for key, val in filters.items():
                if key not in ALLOWED_FILTER_COLUMNS or val is None:
                    continue
                where_clauses.append(f"{key} = %s")
                params.append(val)

        params.extend([q_lit, top_k])
        sql = f"""
        SELECT chunk_id, document_id, source, page, text, metadata,
               1.0 - (embedding <=> %s::vector) AS score
        FROM chunks
        WHERE {' AND '.join(where_clauses)}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """
        results: list[dict[str, Any]] = []
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                for rank, (chunk_id, doc_id, source, page, text, metadata, score) in enumerate(
                    cur.fetchall(), start=1
                ):
                    results.append(
                        {
                            "id": chunk_id,
                            "source_id": doc_id,
                            "source": source,
                            "page": page,
                            "text": text,
                            "metadata": metadata if isinstance(metadata, dict) else {},
                            "score": round(float(score), 6),
                            "rank": rank,
                        }
                    )
        return results


class HybridRRFRetriever:
    """Hybrid retriever combining Dense (pgvector) and BM25 via Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense: DensePgVectorRetriever,
        lexical: BM25Retriever,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.dense = dense
        self.lexical = lexical
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        dense_hits = self.dense.search(query, top_k=self.candidate_k, filters=filters)
        lex_hits = self.lexical.search(query, top_k=self.candidate_k, filters=filters)

        scores: dict[str, float] = {}
        chunks_by_id: dict[str, dict[str, Any]] = {}

        for hit in dense_hits:
            cid = hit["id"]
            chunks_by_id[cid] = hit
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + hit["rank"])

        for hit in lex_hits:
            cid = hit["id"]
            if cid not in chunks_by_id:
                chunks_by_id[cid] = hit
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + hit["rank"])

        ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_k]
        results: list[dict[str, Any]] = []
        for rank, cid in enumerate(ranked_ids, start=1):
            ch = dict(chunks_by_id[cid])
            ch["score"] = round(scores[cid], 6)
            ch["rank"] = rank
            results.append(ch)
        return results
