"""Request/response Pydantic models for the search API."""

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    limit: int = Field(10, ge=1, le=100, description="Max results to return")
    year_min: int | None = Field(None, description="Minimum publication year")
    year_max: int | None = Field(None, description="Maximum publication year")
    min_citations: int | None = Field(None, description="Minimum citation count")
    sections: list[str] | None = Field(None, description="Filter by section types (e.g. ['results', 'methods'])")


class SearchResult(BaseModel):
    paper_id: str
    doi: str = ""
    title: str = ""
    authors: list[str] = []
    year: int | None = None
    score: float = 0.0
    matching_section: str = ""
    matching_text: str = ""
    section_type: str = ""
    citation_count: int = 0
    journal: str = ""
    chunking_method: str = ""


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total: int
    query: str


class PaperChunksResponse(BaseModel):
    paper_id: str
    chunks: list[dict]
    total: int


class StatsResponse(BaseModel):
    points_count: int
    vectors_count: int | None = None
    status: str


class HealthResponse(BaseModel):
    status: str = "ok"
    collection: str = ""
    points_count: int = 0
