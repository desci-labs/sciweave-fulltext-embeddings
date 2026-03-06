# Embedding Evaluation Script

## Context
We have 333 chunks from 4 papers in Qdrant (smoke test). Before scaling to 100K papers, we need to evaluate whether the embeddings are good enough for semantic search. Three evaluation dimensions: retrieval quality, fulltext vs abstract comparison, and intrinsic embedding metrics.

## New File
`scripts/evaluate_embeddings.py` — single self-contained script (~500-600 lines)

## Subcommands
```bash
python scripts/evaluate_embeddings.py retrieval    # Recall@K, MRR
python scripts/evaluate_embeddings.py compare      # fulltext vs abstract-only
python scripts/evaluate_embeddings.py intrinsic    # cosine similarity, clustering
python scripts/evaluate_embeddings.py all          # all three
```

## Design

### 1. Query Generation (no LLM, no external APIs)
Auto-generate benchmark queries from data already in Qdrant:
- **title** (1 per paper): paper title as query — sanity check baseline
- **abstract_sentence** (1 per paper): a 40-200 char sentence from abstract chunk
- **method_phrase** (0-1): sentence from methods/introduction chunks
- **result_phrase** (0-1): sentence from results/conclusion chunks

Sentence quality filter: skip citations `[N]`, formulas `{=}`, URLs. Deterministic via `np.random.default_rng(seed=42)`.

### 2. Retrieval Evaluation
For each query: embed with `BGEEmbedder.embed_query()` → search Qdrant (top_k * 5 over-fetch, paper-level dedup) → check if source paper appears in top-K.

Metrics: **Recall@1, Recall@5, Recall@10, MRR** — overall and broken down by query type.

### 3. Fulltext vs Abstract Comparison
Same queries searched against both:
- Local fulltext: `works_fulltext_embeddings` (localhost:6333)
- Remote abstract: `works_articles_embeddings` (ml-qdrant-2.westeurope.cloudapp.azure.com:6333)

Detect vector dim mismatch via `get_collection()`. If abstract collection uses different dim (not 384), skip comparison gracefully with a note rather than crash.

Metrics: fulltext_recall, abstract_recall, fulltext_wins, abstract_wins, both_found, neither_found.

### 4. Intrinsic Metrics
Fetch vectors from Qdrant (`scroll` with `with_vectors=True`), sample max 50 papers / 20 chunks each.

- **Intra-paper similarity**: avg cosine sim between chunks of same paper (expect 0.6-0.8)
- **Inter-paper similarity**: avg cosine sim between chunks of different papers (expect lower)
- **Separation ratio**: intra/inter (>1.5 is good)
- **Section-type clustering**: avg sim between chunks of same section_type across papers
- **Chunk length stats**: avg, median, p5, p95, by section_type

### 5. Output
- JSON file to `results/eval_TIMESTAMP.json`
- Console summary table printed to stdout

### Key Classes
- `FulltextEvalClient` — direct QdrantClient wrapper (not through FastAPI)
- `AbstractCollectionClient` — remote Qdrant with dim detection
- `QueryGenerator` — auto-generates BenchmarkQuery objects from indexed data
- `RetrievalEvaluator` — Recall@K, MRR computation
- `ComparisonEvaluator` — head-to-head fulltext vs abstract
- `IntrinsicEvaluator` — cosine similarity and clustering metrics

### Files to reuse
- `pipeline/embedder.py`: `BGEEmbedder.embed_query()` for query embedding
- `pipeline/storage.py`: `QdrantClient` init pattern with `check_compatibility=False`
- `scripts/run_pipeline.py`: CLI pattern (sys.path, dotenv, argparse, logging)
- `api/search.py`: paper-level dedup logic (replicate inline)

## Verification
```bash
# With current 4 papers:
python scripts/evaluate_embeddings.py all --output results/eval_smoke.json
# Check results/eval_smoke.json for valid metrics
# Expect: title queries have Recall@1 near 1.0, separation_ratio > 1.0
```
