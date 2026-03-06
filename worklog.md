# Worklog - SciWeave Full-Text Embeddings Pipeline

## Session 1 - 2026-03-06

### Phase 1: Project Scaffolding
- Created directory structure: `pipeline/`, `api/`, `scripts/`, `tests/fixtures/`, `config/`, `.claude-plans/`
- Added `requirements.txt` with all dependencies (elasticsearch, aiohttp, fastembed, qdrant-client, fastapi, openai, etc.)
- Created `.env` with ES, Chutes, GROBID, and Qdrant credentials
- Created `.env.example` (sanitized template)
- Created `.gitignore` (excludes .env, __pycache__, .db, /data/)
- Created `config/default.yaml` with full pipeline configuration
- Copied `PLAN.md` to `.claude-plans/PLAN.md`
- Updated `PLAN.md` with:
  - Hybrid chunking strategy (sentence split for short sections, LLM semantic chunking via Chutes for long sections)
  - 100K paper alpha sample target (sorted by citation count desc)
  - `openai/gpt-oss-120b-TEE` as the semantic chunking model
  - Environment variables from ml-sciweave-backend/.env

### Phase 2: Core Pipeline Modules
- `pipeline/__init__.py` - Shared dataclasses: PaperMetadata, Section, Chunk, DownloadResult
- `pipeline/discovery.py` - PaperDiscovery class: ES scroll queries with V0 filters (OA, English, 2015+, >10 citations), sorted by cited_by_count desc, 100K limit support. Uses `best_locations[*].pdf_url` then `locations[*].pdf_url` for PDF URLs.
- `pipeline/downloader.py` - PDFDownloader class: async aiohttp downloads with per-domain rate limiting, concurrency semaphore, retry with exponential backoff, skip-if-exists logic
- `pipeline/extractor.py` - GROBIDExtractor class: sends PDFs to GROBID, parses TEI XML into structured sections (title, abstract, body sections, references). Section type classification from title keywords.
- `pipeline/chunker.py` - HybridChunker class: Tier 1 sentence splitting for sections ≤3000 chars, Tier 2 LLM semantic chunking via Chutes API (openai/gpt-oss-120b-TEE) for longer sections. Falls back to sentence split on LLM failure. Tracks chunking stats.
- `pipeline/embedder.py` - BGEEmbedder class: BAAI/bge-small-en-v1.5 via fastembed, batch embedding, single query embedding
- `pipeline/storage.py` - QdrantStorage class: collection creation with HNSW config and payload indexes, paper upsert with UUID5 point IDs, paper deletion, collection info
- `pipeline/state.py` - PipelineState class: SQLite checkpointing with stages (discovered→downloaded→extracted→chunked→embedded), batch operations, run tracking, stats
