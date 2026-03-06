"""FastAPI search service for fulltext embeddings."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI

from api.models import (
    HealthResponse,
    PaperChunksResponse,
    SearchRequest,
    SearchResponse,
    StatsResponse,
)
from api.search import get_paper_chunks, get_stats, search_fulltext, get_qdrant

app = FastAPI(
    title="SciWeave Full-Text Search",
    description="Semantic search across full-text academic paper embeddings",
    version="0.1.0",
)


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Semantic full-text search with paper-level deduplication."""
    return search_fulltext(request)


@app.get("/paper/{paper_id}/chunks", response_model=PaperChunksResponse)
async def paper_chunks(paper_id: str):
    """Get all chunks for a specific paper."""
    return get_paper_chunks(paper_id)


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Collection statistics."""
    return get_stats()


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check."""
    try:
        qdrant, collection = get_qdrant()
        info = qdrant.get_collection(collection)
        return HealthResponse(
            status="ok",
            collection=collection,
            points_count=info.points_count,
        )
    except Exception as e:
        return HealthResponse(status=f"error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8506)
