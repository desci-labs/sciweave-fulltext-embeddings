#!/usr/bin/env python3
"""
Embedding Evaluation Script for SciWeave Full-Text Embeddings.

Subcommands:
    retrieval   Measure Recall@K and MRR using auto-generated benchmark queries
    compare     Compare fulltext vs abstract-only collection retrieval
    intrinsic   Compute intra/inter-paper cosine similarity and clustering metrics
    all         Run all three evaluations

Usage:
    python scripts/evaluate_embeddings.py retrieval
    python scripts/evaluate_embeddings.py compare
    python scripts/evaluate_embeddings.py intrinsic
    python scripts/evaluate_embeddings.py all --output results/eval.json
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv

load_dotenv()
from pipeline.embedder import BGEEmbedder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkQuery:
    query_text: str
    source_paper_id: str
    source_paper_title: str
    query_type: str       # title, abstract_sentence, method_phrase, result_phrase
    section_origin: str   # section key

@dataclass
class RetrievalResult:
    query: BenchmarkQuery
    rank_found: int | None  # 1-indexed, None if not in top-K
    top_k_paper_ids: list[str]
    top_k_scores: list[float]

@dataclass
class RetrievalMetrics:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    total_queries: int
    found_count: int
    by_query_type: dict

@dataclass
class CompareMetrics:
    total_queries: int
    k: int
    fulltext_recall_at_k: float
    abstract_recall_at_k: float | None
    fulltext_wins: int | None
    abstract_wins: int | None
    both_found: int | None
    neither_found: int | None
    embedding_model_note: str

@dataclass
class IntrinsicMetrics:
    total_papers: int
    total_chunks: int
    avg_chunk_length: float
    median_chunk_length: float
    p5_chunk_length: float
    p95_chunk_length: float
    intra_paper_similarity: float
    inter_paper_similarity: float
    separation_ratio: float
    section_type_clustering: dict
    chunk_length_by_section_type: dict

@dataclass
class EvalResults:
    run_at: str
    collection_name: str
    collection_points: int
    retrieval: RetrievalMetrics | None = None
    compare: CompareMetrics | None = None
    intrinsic: IntrinsicMetrics | None = None


# ---------------------------------------------------------------------------
# Qdrant clients
# ---------------------------------------------------------------------------

class FulltextEvalClient:
    """Direct Qdrant client for the local fulltext collection."""

    def __init__(self, qdrant_url: str = None, collection_name: str = None):
        self.url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection = collection_name or os.getenv(
            "QDRANT_COLLECTION", "works_fulltext_embeddings"
        )
        self.client = QdrantClient(url=self.url, check_compatibility=False)

    def get_collection_info(self) -> dict:
        info = self.client.get_collection(self.collection)
        return {"points_count": info.points_count, "status": info.status.value}

    def get_all_paper_ids(self) -> list[str]:
        """Scroll entire collection to collect unique paper_ids."""
        paper_ids = set()
        offset = None
        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection,
                limit=1000,
                with_payload=["paper_id"],
                with_vectors=False,
                offset=offset,
            )
            if not results:
                break
            for point in results:
                paper_ids.add(point.payload["paper_id"])
            if next_offset is None:
                break
            offset = next_offset
        return sorted(paper_ids)

    def get_paper_chunks(self, paper_id: str, with_vectors: bool = False) -> list[dict]:
        """Get all chunks for a paper."""
        results, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
            ),
            limit=1000,
            with_payload=True,
            with_vectors=with_vectors,
        )
        out = []
        for r in results:
            item = {"payload": r.payload}
            if with_vectors:
                item["vector"] = np.array(r.vector, dtype=np.float32)
            out.append(item)
        return out

    def get_sample_chunks_with_vectors(
        self, max_papers: int = 50, max_chunks_per_paper: int = 20
    ) -> dict[str, list[dict]]:
        """Return {paper_id: [{vector, payload}, ...]} for a sample."""
        paper_ids = self.get_all_paper_ids()[:max_papers]
        result = {}
        for pid in paper_ids:
            chunks = self.get_paper_chunks(pid, with_vectors=True)
            result[pid] = chunks[:max_chunks_per_paper]
        return result

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """Search with paper-level dedup. Returns [(paper_id, score)]."""
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector.tolist(),
            limit=top_k * 5,
            with_payload=["paper_id"],
        )
        # Paper-level dedup: keep best score per paper
        best: dict[str, float] = {}
        for r in results:
            pid = r.payload["paper_id"]
            if pid not in best or r.score > best[pid]:
                best[pid] = r.score
        ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class AbstractCollectionClient:
    """Client for the remote abstract-only collection."""

    ABSTRACT_URL = "http://ml-qdrant-2.westeurope.cloudapp.azure.com:6333"
    ABSTRACT_COLLECTION = "works_articles_embeddings"

    def __init__(self, qdrant_url: str = None, collection_name: str = None):
        self.url = qdrant_url or self.ABSTRACT_URL
        self.collection = collection_name or self.ABSTRACT_COLLECTION
        self.vector_dim = -1
        self.embedding_model_note = ""
        try:
            self.client = QdrantClient(url=self.url, check_compatibility=False, timeout=10)
            self._detect()
        except Exception as e:
            self.client = None
            self.embedding_model_note = f"Abstract collection unreachable: {e}"
            logger.warning(self.embedding_model_note)

    def _detect(self):
        info = self.client.get_collection(self.collection)
        vc = info.config.params.vectors
        if hasattr(vc, "size"):
            self.vector_dim = vc.size
        elif isinstance(vc, dict):
            first = next(iter(vc.values()))
            self.vector_dim = first.size
        else:
            self.vector_dim = -1

        if self.vector_dim == 384:
            self.embedding_model_note = "Same dimension (384) - direct comparison valid"
        elif self.vector_dim > 0:
            self.embedding_model_note = (
                f"Abstract collection uses {self.vector_dim}-dim vectors vs fulltext 384-dim. "
                "Cannot embed queries with BGE-small for abstract collection. Comparison skipped."
            )
        else:
            self.embedding_model_note = "Could not detect abstract collection vector dimension."

    def can_compare(self) -> bool:
        return self.client is not None and self.vector_dim == 384

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        if not self.can_compare():
            return []
        try:
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector.tolist(),
                limit=top_k * 5,
                with_payload=True,
            )
        except Exception as e:
            logger.warning(f"Abstract search failed: {e}")
            return []

        best: dict[str, float] = {}
        for r in results:
            # Try both field names
            pid = r.payload.get("paper_id") or r.payload.get("work_id", "")
            if not pid:
                continue
            if "/" in pid:
                pid = pid.split("/")[-1]
            if pid not in best or r.score > best[pid]:
                best[pid] = r.score
        ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ---------------------------------------------------------------------------
# Query generator
# ---------------------------------------------------------------------------

class QueryGenerator:
    """Auto-generate benchmark queries from indexed Qdrant data."""

    def __init__(self, client: FulltextEvalClient):
        self.client = client

    def generate(
        self, paper_ids: list[str], queries_per_paper: int = 3, rng_seed: int = 42
    ) -> list[BenchmarkQuery]:
        rng = np.random.default_rng(rng_seed)
        all_queries = []
        for pid in paper_ids:
            queries = self._generate_for_paper(pid, queries_per_paper, rng)
            all_queries.extend(queries)
        logger.info(f"Generated {len(all_queries)} queries from {len(paper_ids)} papers")
        return all_queries

    def _generate_for_paper(
        self, paper_id: str, queries_per_paper: int, rng
    ) -> list[BenchmarkQuery]:
        chunks = self.client.get_paper_chunks(paper_id, with_vectors=False)
        if not chunks:
            return []

        title = chunks[0]["payload"].get("title", "")
        if not title:
            return []

        queries = [BenchmarkQuery(
            query_text=title,
            source_paper_id=paper_id,
            source_paper_title=title,
            query_type="title",
            section_origin="title",
        )]

        if queries_per_paper <= 1:
            return queries

        by_type = defaultdict(list)
        for c in chunks:
            by_type[c["payload"].get("section_type", "")].append(c["payload"])

        # Abstract sentence
        for chunk in by_type.get("abstract", []):
            sent = self._pick_sentence(chunk.get("chunk_text", ""), rng)
            if sent:
                queries.append(BenchmarkQuery(
                    query_text=sent,
                    source_paper_id=paper_id,
                    source_paper_title=title,
                    query_type="abstract_sentence",
                    section_origin="abstract",
                ))
                break

        if len(queries) >= queries_per_paper:
            return queries[:queries_per_paper]

        # Method phrase
        method_types = ("methods", "introduction", "body_section")
        for st in method_types:
            for chunk in by_type.get(st, []):
                sent = self._pick_sentence(chunk.get("chunk_text", ""), rng)
                if sent:
                    queries.append(BenchmarkQuery(
                        query_text=sent,
                        source_paper_id=paper_id,
                        source_paper_title=title,
                        query_type="method_phrase",
                        section_origin=st,
                    ))
                    break
            if len(queries) >= queries_per_paper:
                return queries[:queries_per_paper]

        # Result phrase
        result_types = ("results", "conclusion", "discussion")
        for st in result_types:
            for chunk in by_type.get(st, []):
                sent = self._pick_sentence(chunk.get("chunk_text", ""), rng)
                if sent:
                    queries.append(BenchmarkQuery(
                        query_text=sent,
                        source_paper_id=paper_id,
                        source_paper_title=title,
                        query_type="result_phrase",
                        section_origin=st,
                    ))
                    break
            if len(queries) >= queries_per_paper:
                break

        return queries[:queries_per_paper]

    def _pick_sentence(self, text: str, rng, min_len: int = 40, max_len: int = 200) -> str | None:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        candidates = []
        for s in sentences:
            s = s.strip()
            if len(s) < min_len or len(s) > max_len:
                continue
            # Skip citations, formulas, URLs
            if re.search(r"\[\d+\]", s) or re.search(r"[{}=]", s) or "http" in s:
                continue
            candidates.append(s)
        if not candidates:
            return None
        return str(rng.choice(candidates))


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

class RetrievalEvaluator:
    """Evaluate retrieval quality: Recall@K and MRR."""

    def __init__(self, client: FulltextEvalClient, embedder: BGEEmbedder):
        self.client = client
        self.embedder = embedder

    def evaluate(
        self, queries: list[BenchmarkQuery], top_k: int = 10
    ) -> tuple[list[RetrievalResult], RetrievalMetrics]:
        results = []
        for i, q in enumerate(queries):
            vec = self.embedder.embed_query(q.query_text)
            ranked = self.client.search(vec, top_k=top_k)
            paper_ids = [pid for pid, _ in ranked]
            scores = [s for _, s in ranked]

            rank = None
            if q.source_paper_id in paper_ids:
                rank = paper_ids.index(q.source_paper_id) + 1

            results.append(RetrievalResult(
                query=q, rank_found=rank,
                top_k_paper_ids=paper_ids, top_k_scores=scores,
            ))

            if (i + 1) % 10 == 0:
                logger.info(f"  Retrieval: {i + 1}/{len(queries)} queries done")

        metrics = self._compute_metrics(results, top_k)
        return results, metrics

    def _compute_metrics(self, results: list[RetrievalResult], top_k: int) -> RetrievalMetrics:
        total = len(results)
        if total == 0:
            return RetrievalMetrics(0, 0, 0, 0, 0, 0, {})

        def _calc(subset):
            n = len(subset)
            if n == 0:
                return {"total": 0, "recall_at_1": 0, "recall_at_5": 0, "recall_at_10": 0, "mrr": 0}
            r1 = sum(1 for r in subset if r.rank_found is not None and r.rank_found <= 1) / n
            r5 = sum(1 for r in subset if r.rank_found is not None and r.rank_found <= 5) / n
            r10 = sum(1 for r in subset if r.rank_found is not None and r.rank_found <= min(10, top_k)) / n
            mrr = sum((1.0 / r.rank_found) if r.rank_found else 0 for r in subset) / n
            return {"total": n, "recall_at_1": round(r1, 3), "recall_at_5": round(r5, 3),
                    "recall_at_10": round(r10, 3), "mrr": round(mrr, 3)}

        overall = _calc(results)
        by_type = {}
        types = set(r.query.query_type for r in results)
        for qt in sorted(types):
            subset = [r for r in results if r.query.query_type == qt]
            by_type[qt] = _calc(subset)

        return RetrievalMetrics(
            recall_at_1=overall["recall_at_1"],
            recall_at_5=overall["recall_at_5"],
            recall_at_10=overall["recall_at_10"],
            mrr=overall["mrr"],
            total_queries=total,
            found_count=sum(1 for r in results if r.rank_found is not None),
            by_query_type=by_type,
        )


class ComparisonEvaluator:
    """Compare fulltext vs abstract-only retrieval."""

    def __init__(
        self, fulltext: FulltextEvalClient, abstract: AbstractCollectionClient, embedder: BGEEmbedder
    ):
        self.fulltext = fulltext
        self.abstract = abstract
        self.embedder = embedder

    def evaluate(self, queries: list[BenchmarkQuery], top_k: int = 10) -> CompareMetrics:
        if not self.abstract.can_compare():
            # Still compute fulltext recall
            ft_found = 0
            for q in queries:
                vec = self.embedder.embed_query(q.query_text)
                ranked = self.fulltext.search(vec, top_k=top_k)
                if q.source_paper_id in [pid for pid, _ in ranked]:
                    ft_found += 1
            return CompareMetrics(
                total_queries=len(queries), k=top_k,
                fulltext_recall_at_k=round(ft_found / max(len(queries), 1), 3),
                abstract_recall_at_k=None,
                fulltext_wins=None, abstract_wins=None,
                both_found=None, neither_found=None,
                embedding_model_note=self.abstract.embedding_model_note,
            )

        ft_wins = ab_wins = both = neither = 0
        for i, q in enumerate(queries):
            vec = self.embedder.embed_query(q.query_text)
            ft_ids = [pid for pid, _ in self.fulltext.search(vec, top_k=top_k)]
            ab_ids = [pid for pid, _ in self.abstract.search(vec, top_k=top_k)]
            ft_hit = q.source_paper_id in ft_ids
            ab_hit = q.source_paper_id in ab_ids

            if ft_hit and ab_hit:
                both += 1
            elif ft_hit:
                ft_wins += 1
            elif ab_hit:
                ab_wins += 1
            else:
                neither += 1

            if (i + 1) % 10 == 0:
                logger.info(f"  Compare: {i + 1}/{len(queries)} queries done")

        n = max(len(queries), 1)
        return CompareMetrics(
            total_queries=len(queries), k=top_k,
            fulltext_recall_at_k=round((both + ft_wins) / n, 3),
            abstract_recall_at_k=round((both + ab_wins) / n, 3),
            fulltext_wins=ft_wins, abstract_wins=ab_wins,
            both_found=both, neither_found=neither,
            embedding_model_note=self.abstract.embedding_model_note,
        )


class IntrinsicEvaluator:
    """Compute intrinsic embedding quality metrics."""

    def __init__(self, client: FulltextEvalClient):
        self.client = client

    def evaluate(
        self, max_papers: int = 50, max_chunks_per_paper: int = 20, inter_sample_size: int = 500
    ) -> IntrinsicMetrics:
        logger.info(f"Fetching vectors for up to {max_papers} papers...")
        paper_chunks = self.client.get_sample_chunks_with_vectors(max_papers, max_chunks_per_paper)

        all_lengths = []
        len_by_type: dict[str, list[int]] = defaultdict(list)
        total_chunks = 0
        for pid, chunks in paper_chunks.items():
            for c in chunks:
                total_chunks += 1
                text = c["payload"].get("chunk_text", "")
                l = len(text)
                all_lengths.append(l)
                st = c["payload"].get("section_type", "unknown")
                len_by_type[st].append(l)

        if not all_lengths:
            return IntrinsicMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, {}, {})

        arr = np.array(all_lengths)
        chunk_stats = {
            "avg": round(float(np.mean(arr)), 1),
            "median": round(float(np.median(arr)), 1),
            "p5": round(float(np.percentile(arr, 5)), 1),
            "p95": round(float(np.percentile(arr, 95)), 1),
        }

        len_by_type_avg = {
            st: round(float(np.mean(lens)), 1) for st, lens in sorted(len_by_type.items())
        }

        rng = np.random.default_rng(42)
        intra = self._intra_paper_similarity(paper_chunks)
        inter = self._inter_paper_similarity(paper_chunks, inter_sample_size, rng)
        sep = round(intra / inter, 3) if inter > 0 else float("inf")
        clustering = self._section_clustering(paper_chunks)

        return IntrinsicMetrics(
            total_papers=len(paper_chunks),
            total_chunks=total_chunks,
            avg_chunk_length=chunk_stats["avg"],
            median_chunk_length=chunk_stats["median"],
            p5_chunk_length=chunk_stats["p5"],
            p95_chunk_length=chunk_stats["p95"],
            intra_paper_similarity=intra,
            inter_paper_similarity=inter,
            separation_ratio=sep,
            section_type_clustering=clustering,
            chunk_length_by_section_type=len_by_type_avg,
        )

    def _intra_paper_similarity(self, paper_chunks: dict) -> float:
        sims = []
        for pid, chunks in paper_chunks.items():
            if len(chunks) < 2:
                continue
            vecs = np.array([c["vector"] for c in chunks])
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            vecs_norm = vecs / norms
            sim_matrix = vecs_norm @ vecs_norm.T
            # Upper triangle excluding diagonal
            triu_indices = np.triu_indices(len(chunks), k=1)
            paper_sims = sim_matrix[triu_indices]
            if len(paper_sims) > 0:
                sims.append(float(np.mean(paper_sims)))
        return round(float(np.mean(sims)), 3) if sims else 0.0

    def _inter_paper_similarity(self, paper_chunks: dict, sample_size: int, rng) -> float:
        # Build flat list of (paper_id, vector)
        all_items = []
        for pid, chunks in paper_chunks.items():
            for c in chunks:
                all_items.append((pid, c["vector"]))

        if len(all_items) < 2:
            return 0.0

        sims = []
        attempts = 0
        while len(sims) < sample_size and attempts < sample_size * 10:
            i, j = rng.integers(0, len(all_items), size=2)
            if i == j or all_items[i][0] == all_items[j][0]:
                attempts += 1
                continue
            a, b = all_items[i][1], all_items[j][1]
            norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                attempts += 1
                continue
            sim = float(np.dot(a, b) / (norm_a * norm_b))
            sims.append(sim)
            attempts += 1

        return round(float(np.mean(sims)), 3) if sims else 0.0

    def _section_clustering(self, paper_chunks: dict) -> dict:
        by_type: dict[str, list[np.ndarray]] = defaultdict(list)
        for pid, chunks in paper_chunks.items():
            for c in chunks:
                st = c["payload"].get("section_type", "unknown")
                by_type[st].append(c["vector"])

        result = {}
        for st, vectors in sorted(by_type.items()):
            if len(vectors) < 2:
                continue
            vecs = np.array(vectors)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            vecs_norm = vecs / norms
            sim_matrix = vecs_norm @ vecs_norm.T
            triu = np.triu_indices(len(vectors), k=1)
            result[st] = round(float(np.mean(sim_matrix[triu])), 3)

        return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(results: EvalResults):
    print(f"\n{'=' * 50}")
    print(f"  SciWeave Embedding Evaluation")
    print(f"{'=' * 50}")
    print(f"  Run at     : {results.run_at}")
    print(f"  Collection : {results.collection_name} ({results.collection_points} points)")

    if results.retrieval:
        r = results.retrieval
        print(f"\n--- Retrieval Quality ---")
        print(f"  Queries : {r.total_queries} ({r.found_count} found in top-K)")
        print(f"  Recall@1  : {r.recall_at_1:.3f}")
        print(f"  Recall@5  : {r.recall_at_5:.3f}")
        print(f"  Recall@10 : {r.recall_at_10:.3f}")
        print(f"  MRR       : {r.mrr:.3f}")
        if r.by_query_type:
            print(f"\n  By query type:")
            for qt, m in r.by_query_type.items():
                print(f"    {qt:20s}  R@1={m['recall_at_1']:.2f}  R@10={m['recall_at_10']:.2f}  MRR={m['mrr']:.2f}  (n={m['total']})")

    if results.compare:
        c = results.compare
        print(f"\n--- Fulltext vs Abstract Comparison ---")
        if c.abstract_recall_at_k is not None:
            print(f"  Fulltext Recall@{c.k} : {c.fulltext_recall_at_k:.3f}")
            print(f"  Abstract Recall@{c.k} : {c.abstract_recall_at_k:.3f}")
            print(f"  Fulltext wins : {c.fulltext_wins}  |  Abstract wins : {c.abstract_wins}")
            print(f"  Both found    : {c.both_found}  |  Neither found : {c.neither_found}")
        else:
            print(f"  [SKIPPED] {c.embedding_model_note}")
            print(f"  Fulltext Recall@{c.k} : {c.fulltext_recall_at_k:.3f}")

    if results.intrinsic:
        m = results.intrinsic
        print(f"\n--- Intrinsic Metrics ---")
        print(f"  Papers : {m.total_papers}  |  Chunks : {m.total_chunks}")
        print(f"  Chunk length : avg={m.avg_chunk_length:.0f}  median={m.median_chunk_length:.0f}  p5={m.p5_chunk_length:.0f}  p95={m.p95_chunk_length:.0f}")
        print(f"  Intra-paper sim  : {m.intra_paper_similarity:.3f}")
        print(f"  Inter-paper sim  : {m.inter_paper_similarity:.3f}")
        print(f"  Separation ratio : {m.separation_ratio:.3f}  {'(good)' if m.separation_ratio > 1.5 else '(low)' if m.separation_ratio < 1.2 else ''}")
        if m.section_type_clustering:
            print(f"\n  Section-type clustering:")
            for st, sim in sorted(m.section_type_clustering.items(), key=lambda x: -x[1]):
                print(f"    {st:20s} : {sim:.3f}")

    print()


def save_results(results: EvalResults, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data = asdict(results)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Results saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SciWeave Embedding Evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared args
    for name in ("retrieval", "compare", "intrinsic", "all"):
        p = sub.add_parser(name)
        p.add_argument("--qdrant-url", type=str, default=None)
        p.add_argument("--collection", type=str, default=None)
        p.add_argument("--output", type=str, default=None)
        p.add_argument("--verbose", action="store_true")
        p.add_argument("--seed", type=int, default=42)

        if name in ("retrieval", "all"):
            p.add_argument("--queries-per-paper", type=int, default=3)
            p.add_argument("--top-k", type=int, default=10)

        if name in ("compare", "all"):
            p.add_argument("--abstract-url", type=str, default=None)
            p.add_argument("--abstract-collection", type=str, default=None)
            if name == "compare":
                p.add_argument("--top-k", type=int, default=10)
                p.add_argument("--queries-per-paper", type=int, default=3)

        if name in ("intrinsic", "all"):
            p.add_argument("--max-papers", type=int, default=50)
            p.add_argument("--max-chunks", type=int, default=20)
            p.add_argument("--inter-samples", type=int, default=500)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    cmd = args.command
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output_path = args.output or f"results/eval_{ts}.json"

    fulltext = FulltextEvalClient(qdrant_url=args.qdrant_url, collection_name=args.collection)
    info = fulltext.get_collection_info()
    logger.info(f"Collection: {fulltext.collection} ({info['points_count']} points)")

    results = EvalResults(
        run_at=datetime.now(timezone.utc).isoformat(),
        collection_name=fulltext.collection,
        collection_points=info["points_count"],
    )

    # Generate queries if needed
    queries = None
    embedder = None
    if cmd in ("retrieval", "compare", "all"):
        embedder = BGEEmbedder()
        paper_ids = fulltext.get_all_paper_ids()
        qpp = getattr(args, "queries_per_paper", 3)
        generator = QueryGenerator(fulltext)
        queries = generator.generate(paper_ids, queries_per_paper=qpp, rng_seed=args.seed)

    if cmd in ("retrieval", "all"):
        logger.info("Running retrieval evaluation...")
        evaluator = RetrievalEvaluator(fulltext, embedder)
        top_k = getattr(args, "top_k", 10)
        _, results.retrieval = evaluator.evaluate(queries, top_k=top_k)

    if cmd in ("compare", "all"):
        logger.info("Running fulltext vs abstract comparison...")
        ab_url = getattr(args, "abstract_url", None)
        ab_col = getattr(args, "abstract_collection", None)
        abstract = AbstractCollectionClient(qdrant_url=ab_url, collection_name=ab_col)
        top_k = getattr(args, "top_k", 10)
        comp = ComparisonEvaluator(fulltext, abstract, embedder)
        results.compare = comp.evaluate(queries, top_k=top_k)

    if cmd in ("intrinsic", "all"):
        logger.info("Running intrinsic metrics evaluation...")
        max_p = getattr(args, "max_papers", 50)
        max_c = getattr(args, "max_chunks", 20)
        inter_s = getattr(args, "inter_samples", 500)
        intr = IntrinsicEvaluator(fulltext)
        results.intrinsic = intr.evaluate(max_papers=max_p, max_chunks_per_paper=max_c, inter_sample_size=inter_s)

    print_summary(results)
    save_results(results, output_path)


if __name__ == "__main__":
    main()
