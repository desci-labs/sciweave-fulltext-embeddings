"""Search endpoint handlers."""

import logging
import os
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    Range,
)

from api.models import (
    PaperChunksResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    StatsResponse,
)
from pipeline.embedder import BGEEmbedder

logger = logging.getLogger(__name__)

# Module-level singletons (initialized once)
_embedder: BGEEmbedder | None = None
_qdrant: QdrantClient | None = None
_collection_name: str = ""


def get_embedder() -> BGEEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = BGEEmbedder()
    return _embedder


def get_qdrant() -> tuple[QdrantClient, str]:
    global _qdrant, _collection_name
    if _qdrant is None:
        _qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        _collection_name = os.getenv("QDRANT_COLLECTION", "works_fulltext_embeddings")
    return _qdrant, _collection_name


def search_fulltext(request: SearchRequest) -> SearchResponse:
    """Semantic search across full-text paper embeddings with paper-level dedup."""
    embedder = get_embedder()
    qdrant, collection = get_qdrant()

    # Embed query
    query_vector = embedder.embed_query(request.query)

    # Build filters
    query_filter = _build_filters(request)

    # Over-fetch for deduplication
    fetch_limit = request.limit * 5

    results = qdrant.search(
        collection_name=collection,
        query_vector=query_vector.tolist(),
        query_filter=query_filter,
        limit=fetch_limit,
        with_payload=True,
    )

    # Deduplicate by paper
    deduplicated = _deduplicate_by_paper(
        results,
        max_results=request.limit,
        include_context=request.include_context,
        max_chunks_per_paper=request.max_chunks_per_paper,
    )

    return SearchResponse(
        results=deduplicated,
        total=len(deduplicated),
        query=request.query,
    )


def get_paper_chunks(paper_id: str) -> PaperChunksResponse:
    """Get all chunks for a specific paper."""
    qdrant, collection = get_qdrant()

    results, _ = qdrant.scroll(
        collection_name=collection,
        scroll_filter=Filter(
            must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
        ),
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )

    chunks = sorted(
        [r.payload for r in results],
        key=lambda x: (x.get("section_order", 0), x.get("chunk_index", 0)),
    )

    return PaperChunksResponse(
        paper_id=paper_id,
        chunks=chunks,
        total=len(chunks),
    )


def get_stats() -> StatsResponse:
    """Get collection statistics."""
    qdrant, collection = get_qdrant()
    info = qdrant.get_collection(collection)
    return StatsResponse(
        points_count=info.points_count,
        vectors_count=info.vectors_count,
        status=info.status.value,
    )


def _build_filters(request: SearchRequest) -> Filter | None:
    """Build Qdrant filter from search request."""
    conditions = []

    if request.year_min is not None:
        conditions.append(
            FieldCondition(key="year", range=Range(gte=request.year_min))
        )
    if request.year_max is not None:
        conditions.append(
            FieldCondition(key="year", range=Range(lte=request.year_max))
        )
    if request.min_citations is not None:
        conditions.append(
            FieldCondition(key="citation_count", range=Range(gte=request.min_citations))
        )
    if request.sections:
        # Match any of the specified sections
        should = [
            FieldCondition(key="section_type", match=MatchValue(value=s))
            for s in request.sections
        ]
        conditions.append(Filter(should=should))

    if not conditions:
        return None

    return Filter(must=conditions)


def _deduplicate_by_paper(
    results,
    max_results: int,
    include_context: bool = False,
    max_chunks_per_paper: int = 3,
) -> list[SearchResult]:
    """Deduplicate search results at the paper level.

    Groups by paper_id, keeps the highest-scoring chunk per paper.
    If include_context=True, returns up to max_chunks_per_paper per paper.
    """
    paper_groups: dict[str, list] = defaultdict(list)

    for result in results:
        payload = result.payload
        paper_id = payload.get("paper_id", "")
        paper_groups[paper_id].append((result.score, payload))

    deduplicated = []
    for paper_id, chunks in paper_groups.items():
        # Sort chunks by score desc
        chunks.sort(key=lambda x: x[0], reverse=True)
        best_score, best_payload = chunks[0]

        deduplicated.append(SearchResult(
            paper_id=paper_id,
            doi=best_payload.get("doi", ""),
            title=best_payload.get("title", ""),
            authors=best_payload.get("authors", []),
            year=best_payload.get("year"),
            score=best_score,
            matching_section=best_payload.get("section", ""),
            matching_text=best_payload.get("chunk_text", ""),
            section_type=best_payload.get("section_type", ""),
            citation_count=best_payload.get("citation_count", 0),
            journal=best_payload.get("journal", ""),
            chunking_method=best_payload.get("chunking_method", ""),
        ))

    # Sort by score and limit
    deduplicated.sort(key=lambda x: x.score, reverse=True)
    return deduplicated[:max_results]
