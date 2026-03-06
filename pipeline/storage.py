"""Qdrant storage client for fulltext embeddings."""

import logging
import os
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from pipeline import Chunk, PaperMetadata

logger = logging.getLogger(__name__)


class QdrantStorage:
    """Manage Qdrant collection for fulltext embeddings."""

    def __init__(
        self,
        qdrant_url: str = None,
        collection_name: str = None,
    ):
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection_name = collection_name or os.getenv(
            "QDRANT_COLLECTION", "works_fulltext_embeddings"
        )
        self.client = QdrantClient(url=self.qdrant_url)

    def ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            logger.info(f"Collection '{self.collection_name}' already exists")
            return

        logger.info(f"Creating collection '{self.collection_name}'")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE,
                on_disk=True,
            ),
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100,
                on_disk=True,
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000,
                memmap_threshold=50000,
            ),
            on_disk_payload=True,
        )

        # Create payload indexes
        indexes = [
            ("paper_id", PayloadSchemaType.KEYWORD),
            ("year", PayloadSchemaType.INTEGER),
            ("section", PayloadSchemaType.KEYWORD),
            ("section_type", PayloadSchemaType.KEYWORD),
            ("citation_count", PayloadSchemaType.INTEGER),
        ]
        for field_name, schema_type in indexes:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )

        logger.info(f"Collection '{self.collection_name}' created with indexes")

    def upsert_paper(
        self,
        paper_id: str,
        chunks: list[Chunk],
        embeddings: list[np.ndarray],
        metadata: PaperMetadata,
    ):
        """Upsert all chunks for a single paper."""
        if not chunks or not embeddings:
            return

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{paper_id}:{chunk.section}:{chunk.chunk_index}"))
            points.append(PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "paper_id": paper_id,
                    "doi": metadata.doi or "",
                    "title": metadata.title,
                    "authors": metadata.authors,
                    "year": metadata.year,
                    "citation_count": metadata.citation_count,
                    "journal": metadata.journal or "",
                    "chunk_text": chunk.text,
                    "section": chunk.section,
                    "section_type": chunk.section_type,
                    "section_order": chunk.section_order,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "chunking_method": chunk.chunking_method,
                },
            ))

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

    def delete_paper(self, paper_id: str):
        """Delete all chunks for a paper (for re-processing)."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
            ),
        )

    def get_collection_info(self) -> dict:
        """Get collection stats."""
        info = self.client.get_collection(self.collection_name)
        return {
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status.value,
        }
