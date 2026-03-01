# SciWeave Full-Text Embeddings Pipeline

## Overview

A standalone pipeline that discovers open-access academic papers, downloads their PDFs, extracts full text via GROBID, chunks sections into embedding-ready segments, generates vector embeddings, and stores them in a dedicated Qdrant collection for semantic full-text search.

This is an **independent repository** with no code dependencies on `ml-sciweave-backend`. It runs as its own service with its own API.

---

## V0 Scope

- **Papers**: Open Access with available PDFs
- **Date range**: 2015 onwards
- **Language**: English only
- **Citation filter**: >10 citations (quality signal)
- **Estimated corpus**: ~8-12M papers
- **Estimated chunks**: ~240M (avg ~20-30 chunks/paper)
- **Estimated storage**: ~528 GB in Qdrant (384-dim vectors + payloads)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Orchestrator                                 │
│   (manages state, retries, progress tracking, rate limiting)        │
└──────┬──────────┬────────────┬────────────┬────────────┬────────────┘
       │          │            │            │            │
       v          v            v            v            v
  ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌─────────┐
  │Discovery│ │Downloader│ │Extractor │ │ Chunker │ │Embedder │
  │  (ES)   │ │  (HTTP)  │ │ (GROBID) │ │(section)│ │(BGE-sm) │
  └────┬────┘ └────┬─────┘ └────┬─────┘ └────┬────┘ └────┬────┘
       │           │            │            │            │
       v           v            v            v            v
  ┌─────────────────────────────────────────────────────────────┐
  │                    Storage (Qdrant)                          │
  │            Collection: works_fulltext_embeddings             │
  └─────────────────────────────────────────────────────────────┘
                              │
                              v
                    ┌───────────────────┐
                    │   Search API      │
                    │   (FastAPI)       │
                    └───────────────────┘
```

### Pipeline Stages

1. **Discovery** - Query Elasticsearch (OpenAlex or similar) for OA papers matching V0 criteria. Yields batches of paper metadata (DOI, PDF URL, title, authors, year, citation count).

2. **Downloader** - Fetches PDFs from discovered URLs. Handles redirects, retries, rate limiting per domain. Stores raw PDFs temporarily on disk.

3. **Extractor** - Sends PDFs to a self-hosted GROBID instance. Parses TEI XML into structured sections (title, abstract, introduction, methods, results, etc.).

4. **Chunker** - Section-aware chunking:
   - Preserves section boundaries (never splits across sections)
   - Max chunk size: 1000 chars
   - Splits on sentence boundaries within sections
   - Attaches metadata: section name, section type, order, paper ID

5. **Embedder** - Generates embeddings using `BAAI/bge-small-en-v1.5` (384 dimensions). Batched processing (batch size 256). CPU or GPU.

6. **Storage** - Upserts embedding points into Qdrant with full payload metadata.

---

## Repository Structure

```
sciweave-fulltext-embeddings/
├── PLAN.md
├── README.md
├── requirements.txt
├── .env.example
├── docker-compose.yml          # GROBID + Qdrant + pipeline
│
├── pipeline/
│   ├── __init__.py
│   ├── discovery.py            # ES query builder, batch iterator
│   ├── downloader.py           # Async PDF downloader with rate limiting
│   ├── extractor.py            # GROBID client, TEI parser
│   ├── chunker.py              # Section-aware text chunking
│   ├── embedder.py             # BGE-small embedding generation
│   ├── storage.py              # Qdrant upsert client
│   ├── orchestrator.py         # Pipeline coordinator, state machine
│   └── state.py                # Progress tracking, checkpointing (SQLite)
│
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app
│   ├── search.py               # Search endpoint handlers
│   └── models.py               # Request/response Pydantic models
│
├── scripts/
│   ├── run_pipeline.py         # CLI entry point for batch processing
│   ├── benchmark_search.py     # Compare fulltext vs abstract-only retrieval
│   ├── check_progress.py       # Print pipeline progress stats
│   └── backfill.py             # Re-process failed papers
│
├── tests/
│   ├── test_chunker.py
│   ├── test_extractor.py
│   ├── test_embedder.py
│   ├── test_search.py
│   └── fixtures/               # Sample PDFs, TEI XML, etc.
│
└── config/
    ├── default.yaml            # Default pipeline configuration
    └── production.yaml         # Production overrides
```

---

## Module Designs

### `pipeline/discovery.py`

```python
class PaperDiscovery:
    """Query Elasticsearch for OA papers matching V0 criteria."""

    def __init__(self, es_url: str, index: str, batch_size: int = 500):
        ...

    def build_query(self, min_year: int = 2015, min_citations: int = 10,
                    language: str = "en", must_have_pdf: bool = True) -> dict:
        """Build ES query for V0 criteria."""
        ...

    def iterate_batches(self, query: dict) -> Iterator[List[PaperMetadata]]:
        """Scroll through results yielding batches of paper metadata."""
        ...

    def estimate_total(self, query: dict) -> int:
        """Count total matching papers without fetching."""
        ...
```

### `pipeline/downloader.py`

```python
class PDFDownloader:
    """Async PDF downloader with per-domain rate limiting."""

    def __init__(self, output_dir: str, max_concurrent: int = 10,
                 rate_limit_per_domain: float = 1.0):
        ...

    async def download_batch(self, papers: List[PaperMetadata]) -> List[DownloadResult]:
        """Download PDFs for a batch of papers. Returns results with status."""
        ...

    async def _download_single(self, paper: PaperMetadata) -> DownloadResult:
        """Download single PDF with retries and redirect handling."""
        ...
```

### `pipeline/extractor.py`

```python
class GROBIDExtractor:
    """Extract structured text from PDFs using GROBID."""

    def __init__(self, grobid_url: str = "http://localhost:8070"):
        ...

    def extract(self, pdf_path: str) -> ExtractedPaper:
        """Send PDF to GROBID, parse TEI XML into structured sections."""
        ...

    def _parse_tei(self, tei_xml: str) -> Dict[str, Section]:
        """Parse TEI XML into section dict."""
        ...
```

### `pipeline/chunker.py`

```python
class SectionAwareChunker:
    """Chunk paper sections preserving section boundaries."""

    def __init__(self, max_chunk_size: int = 1000):
        ...

    def chunk_paper(self, paper_id: str, sections: Dict[str, Section]) -> List[Chunk]:
        """Chunk all sections of a paper."""
        ...

    def _chunk_section(self, section: Section) -> List[str]:
        """Split section text on sentence boundaries."""
        ...
```

### `pipeline/embedder.py`

```python
class BGEEmbedder:
    """Generate embeddings using BAAI/bge-small-en-v1.5."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5",
                 batch_size: int = 256, device: str = "cpu"):
        ...

    def embed_chunks(self, chunks: List[Chunk]) -> List[np.ndarray]:
        """Generate embeddings for a list of chunks."""
        ...
```

### `pipeline/storage.py`

```python
class QdrantStorage:
    """Manage Qdrant collection for fulltext embeddings."""

    COLLECTION_NAME = "works_fulltext_embeddings"

    def __init__(self, qdrant_url: str, qdrant_api_key: str = None):
        ...

    def ensure_collection(self):
        """Create collection if it doesn't exist."""
        ...

    def upsert_paper(self, paper_id: str, chunks: List[Chunk],
                     embeddings: List[np.ndarray], metadata: PaperMetadata):
        """Upsert all chunks for a single paper."""
        ...

    def delete_paper(self, paper_id: str):
        """Delete all chunks for a paper (for re-processing)."""
        ...
```

### `pipeline/state.py`

```python
class PipelineState:
    """Track pipeline progress using SQLite for checkpointing."""

    def __init__(self, db_path: str = "pipeline_state.db"):
        ...

    def mark_discovered(self, paper_id: str, metadata: dict): ...
    def mark_downloaded(self, paper_id: str, pdf_path: str): ...
    def mark_extracted(self, paper_id: str, num_sections: int): ...
    def mark_embedded(self, paper_id: str, num_chunks: int): ...
    def mark_failed(self, paper_id: str, stage: str, error: str): ...
    def get_pending(self, stage: str, limit: int = 500) -> List[str]: ...
    def get_stats(self) -> dict: ...
```

### `pipeline/orchestrator.py`

```python
class PipelineOrchestrator:
    """Coordinate the full pipeline with checkpointing and error handling."""

    def __init__(self, config: dict):
        self.discovery = PaperDiscovery(...)
        self.downloader = PDFDownloader(...)
        self.extractor = GROBIDExtractor(...)
        self.chunker = SectionAwareChunker(...)
        self.embedder = BGEEmbedder(...)
        self.storage = QdrantStorage(...)
        self.state = PipelineState(...)

    def run(self, batch_size: int = 500, max_papers: int = None):
        """Run the full pipeline with progress tracking."""
        ...

    def resume(self):
        """Resume from last checkpoint."""
        ...
```

---

## Qdrant Collection Design

### Collection: `works_fulltext_embeddings`

```python
collection_config = {
    "vectors_config": VectorParams(
        size=384,                    # BGE-small-en-v1.5 dimension
        distance=Distance.COSINE,
        on_disk=True,                # Vectors on disk (528 GB won't fit in RAM)
    ),
    "hnsw_config": HnswConfigDiff(
        m=16,                        # HNSW connections per node
        ef_construct=100,            # Construction quality
        on_disk=True,                # HNSW index on disk
    ),
    "optimizers_config": OptimizersConfigDiff(
        indexing_threshold=20000,    # Build index after 20K points
        memmap_threshold=50000,
    ),
    "on_disk_payload": True,         # Payloads on disk too
}
```

### Point Payload Schema

```json
{
    "paper_id": "W2100837269",
    "doi": "10.1234/example",
    "title": "Paper Title",
    "authors": ["Author A", "Author B"],
    "year": 2023,
    "citation_count": 45,
    "journal": "Nature",
    "chunk_text": "The actual text content of this chunk...",
    "section": "results",
    "section_type": "body_section",
    "section_order": 5,
    "chunk_index": 0,
    "chunk_num": 1,
    "total_chunks": 24
}
```

### Payload Indexes

```python
# Indexed fields for filtering
payload_indexes = [
    ("paper_id", PayloadSchemaType.KEYWORD),    # Paper-level dedup in search
    ("year", PayloadSchemaType.INTEGER),         # Date range filtering
    ("section", PayloadSchemaType.KEYWORD),      # Section-specific search
    ("section_type", PayloadSchemaType.KEYWORD), # Filter by section type
    ("citation_count", PayloadSchemaType.INTEGER),# Quality filtering
]
```

---

## Search API

### `api/search.py`

```python
@router.post("/search")
async def search_fulltext(request: SearchRequest) -> SearchResponse:
    """
    Semantic search across full-text paper embeddings.
    Returns deduplicated results at the paper level.
    """
    # 1. Embed query
    query_embedding = embedder.embed_query(request.query)

    # 2. Search Qdrant with optional filters
    filters = build_filters(
        year_min=request.year_min,
        year_max=request.year_max,
        min_citations=request.min_citations,
        sections=request.sections,  # e.g., ["results", "methods"]
    )

    results = qdrant_client.search(
        collection_name="works_fulltext_embeddings",
        query_vector=query_embedding,
        query_filter=filters,
        limit=request.limit * 3,  # Over-fetch for dedup
        with_payload=True,
    )

    # 3. Paper-level deduplication
    # Group by paper_id, keep highest-scoring chunk per paper
    deduplicated = deduplicate_by_paper(results, max_results=request.limit)

    # 4. Return results
    return SearchResponse(
        results=[
            SearchResult(
                paper_id=r.paper_id,
                doi=r.doi,
                title=r.title,
                authors=r.authors,
                year=r.year,
                score=r.score,
                matching_section=r.section,
                matching_text=r.chunk_text,
                citation_count=r.citation_count,
            )
            for r in deduplicated
        ],
        total=len(deduplicated),
        query=request.query,
    )
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/search` | Semantic full-text search with paper-level dedup |
| POST | `/search/multi` | Multi-query search (batch) |
| GET | `/paper/{paper_id}/chunks` | Get all chunks for a paper |
| GET | `/stats` | Collection statistics |
| GET | `/health` | Health check |

### Paper-Level Deduplication Strategy

When searching, multiple chunks from the same paper may match. The dedup strategy:

1. Over-fetch by 3x the requested limit
2. Group results by `paper_id`
3. For each paper, keep the chunk with the highest similarity score
4. Optionally include top-N chunks per paper if `include_context=true`
5. Sort final results by score, return top `limit`

---

## Cost Estimates

### One-Time Compute (~$5-7K)

| Component | Cost | Notes |
|-----------|------|-------|
| GPU instance (embedding) | ~$3-4K | A100 80GB, ~2-3 weeks for 240M chunks |
| GROBID processing | ~$1-2K | CPU instances, ~10M PDFs |
| PDF downloads | ~$500 | Bandwidth, ~10TB total |
| Elasticsearch queries | ~$200 | Discovery phase |

### Monthly Operating Costs (~$460/mo)

| Component | Cost/mo | Notes |
|-----------|---------|-------|
| Qdrant Cloud (528 GB) | ~$350 | On-disk vectors, 1 node |
| GROBID instance (incremental) | ~$50 | Process new papers |
| Search API hosting | ~$30 | Small FastAPI instance |
| Monitoring/logging | ~$30 | Datadog or similar |

### Storage Breakdown

- **240M chunks** x 384 dims x 4 bytes = ~350 GB (vectors)
- **240M chunks** x ~750 bytes avg payload = ~178 GB (payloads)
- **Total**: ~528 GB
- **HNSW index overhead**: ~50-80 GB additional

---

## Testing Strategy

### Unit Tests

- `test_chunker.py`: Verify section boundary preservation, max chunk size, sentence splitting
- `test_extractor.py`: TEI XML parsing with sample fixtures
- `test_embedder.py`: Embedding dimensions, batch processing, determinism

### Integration Tests

- `test_search.py`: End-to-end search with a small test collection (~100 papers)
- Verify paper-level deduplication
- Verify filter combinations (year, citations, section)

### Validation Benchmark

**Fulltext vs Abstract-Only Retrieval**

Compare search quality using a curated set of ~200 queries:

1. Generate query set from paper titles, research questions, method names
2. Search both `works_fulltext_embeddings` and abstract-only collection
3. Measure:
   - **Recall@10**: Does the correct paper appear in top 10?
   - **MRR**: Mean Reciprocal Rank of correct paper
   - **Section diversity**: Does fulltext find papers that abstract-only misses?
4. Expected improvement: fulltext should find ~30-40% more relevant papers for method-specific and result-specific queries

### Benchmark Script

```bash
python scripts/benchmark_search.py \
    --queries fixtures/benchmark_queries.json \
    --fulltext-collection works_fulltext_embeddings \
    --abstract-collection works_embeddings \
    --output results/benchmark_results.json
```

---

## Configuration

### `config/default.yaml`

```yaml
discovery:
  es_url: "http://localhost:9200"
  index: "works"
  batch_size: 500
  min_year: 2015
  min_citations: 10
  language: "en"

downloader:
  output_dir: "/data/pdfs"
  max_concurrent: 10
  rate_limit_per_domain: 1.0  # seconds between requests
  timeout: 30
  max_retries: 3

extractor:
  grobid_url: "http://localhost:8070"
  timeout: 60
  max_concurrent: 5

chunker:
  max_chunk_size: 1000
  overlap: 0  # No overlap for V0

embedder:
  model_name: "BAAI/bge-small-en-v1.5"
  batch_size: 256
  device: "cuda"  # or "cpu"

storage:
  qdrant_url: "http://localhost:6333"
  collection_name: "works_fulltext_embeddings"

pipeline:
  checkpoint_interval: 100  # Save state every N papers
  max_failures_per_batch: 50  # Skip batch if too many failures
  log_level: "INFO"
```

---

## Implementation Phases

### Phase 1: Core Pipeline (Week 1-2)
- Set up repo structure
- Implement `discovery.py`, `downloader.py`, `extractor.py`
- Implement `chunker.py`, `embedder.py`, `storage.py`
- Implement `state.py` for checkpointing
- Unit tests for each module

### Phase 2: Orchestrator + CLI (Week 2-3)
- Implement `orchestrator.py`
- Build `scripts/run_pipeline.py` CLI
- Docker compose for GROBID + Qdrant
- End-to-end test with 100 papers

### Phase 3: Search API (Week 3)
- FastAPI search service
- Paper-level deduplication
- Filter support (year, citations, section)
- Integration tests

### Phase 4: Scale Run (Week 3-5)
- Run on full V0 corpus (~8-12M papers)
- Monitor progress, fix failures
- Optimize batch sizes and concurrency

### Phase 5: Validation (Week 5-6)
- Run benchmark: fulltext vs abstract-only
- Analyze results, tune if needed
- Document findings

---

## Dependencies

```
# requirements.txt
elasticsearch>=8.0.0
aiohttp>=3.9.0
fastembed>=0.7.0
qdrant-client>=1.9.0
fastapi>=0.110.0
uvicorn>=0.29.0
pydantic>=2.0.0
requests>=2.31.0
pyyaml>=6.0
tqdm>=4.66.0
numpy>=1.26.0
```

GROBID runs as a separate Docker container (not a Python dependency).
