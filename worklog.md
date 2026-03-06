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
