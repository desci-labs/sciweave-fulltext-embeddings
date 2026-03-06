# SciWeave Full-Text Embeddings Pipeline

A standalone pipeline that discovers open-access academic papers, downloads their PDFs, extracts full text via GROBID, chunks sections, generates vector embeddings, and stores them in Qdrant for semantic full-text search.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in environment variables
cp .env.example .env

# 3. Start GROBID + Qdrant
docker compose up -d

# 4. Run pipeline (10 papers for smoke test)
python scripts/run_pipeline.py --max-papers 10

# 5. Evaluate embeddings
python scripts/evaluate_embeddings.py all --output results/eval.json

# 6. Start search API
python api/main.py
```

## Architecture

```
ES (discovery) → PDF download → GROBID (extraction) → Chunker → BGE-small (embedding) → Qdrant (storage)
```

### Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| Discovery | `pipeline/discovery.py` | Queries Elasticsearch for OA papers (2015+, >10 citations, English) |
| Download | `pipeline/downloader.py` | Async PDF downloads with per-domain rate limiting |
| Extraction | `pipeline/extractor.py` | GROBID TEI XML parsing into structured sections |
| Chunking | `pipeline/chunker.py` | Sentence-boundary splitting (max 1000 chars per chunk) |
| Embedding | `pipeline/embedder.py` | BAAI/bge-small-en-v1.5 (384 dims) via fastembed |
| Storage | `pipeline/storage.py` | Qdrant with HNSW index, on-disk vectors and payloads |
| State | `pipeline/state.py` | SQLite checkpointing for resume support |

### V0-Alpha Target

100K papers sorted by citation count (highest quality first) as a validation sample before scaling to the full 5M+ paper corpus.

## CLI Reference

### Pipeline

```bash
# Full run (default: 100K papers)
python scripts/run_pipeline.py --max-papers 100000

# Resume from checkpoint
python scripts/run_pipeline.py --resume

# Check progress
python scripts/check_progress.py
```

### Evaluation

```bash
# All evaluations
python scripts/evaluate_embeddings.py all

# Individual evaluations
python scripts/evaluate_embeddings.py retrieval    # Recall@K, MRR
python scripts/evaluate_embeddings.py compare      # fulltext vs abstract-only
python scripts/evaluate_embeddings.py intrinsic    # cosine similarity, clustering
```

### Search API

```bash
python api/main.py  # Starts on port 8506
```

**Endpoints:**
- `POST /search` — Semantic full-text search with paper-level deduplication
- `GET /paper/{paper_id}/chunks` — All chunks for a paper
- `GET /stats` — Collection statistics
- `GET /health` — Health check

## Configuration

Environment variables (see `.env.example`):

| Variable | Description |
|----------|-------------|
| `ES_HOST` | Elasticsearch URL |
| `ES_USER` / `ES_PWD` | ES credentials |
| `ES_INDEX` | ES index name (default: `works_open_05_07`) |
| `GROBID_URL` | GROBID service URL |
| `QDRANT_URL` | Qdrant URL |
| `QDRANT_COLLECTION` | Qdrant collection name |
| `QDRANT_DATA_DIR` | Host path for Qdrant data volume |

## Project Structure

```
pipeline/           Core pipeline modules
api/                FastAPI search service
scripts/            CLI entry points (run, evaluate, check progress)
config/             Pipeline configuration
data/               Downloaded PDFs and Qdrant storage (gitignored)
results/            Evaluation results
```
