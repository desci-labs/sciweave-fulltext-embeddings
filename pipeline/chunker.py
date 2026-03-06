"""Section-aware text chunker using sentence-boundary splitting."""

import logging
import re

from pipeline import Chunk, Section

logger = logging.getLogger(__name__)


class SectionChunker:
    """Chunk paper sections on sentence boundaries, preserving section structure."""

    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size
        self.stats = {"sections_chunked": 0, "total_chunks": 0}

    def chunk_paper(self, paper_id: str, sections: dict[str, Section]) -> list[Chunk]:
        """Chunk all sections of a paper."""
        all_chunks: list[Chunk] = []

        for section_key, section in sections.items():
            if not section.content.strip():
                continue

            chunk_texts = self._sentence_split(section.content.strip())
            self.stats["sections_chunked"] += 1
            self.stats["total_chunks"] += len(chunk_texts)

            for i, text in enumerate(chunk_texts):
                all_chunks.append(Chunk(
                    paper_id=paper_id,
                    text=text,
                    section=section_key,
                    section_type=section.section_type,
                    section_order=section.order,
                    chunk_index=i,
                    total_chunks=len(chunk_texts),
                    chunking_method="sentence_split",
                ))

        return all_chunks

    def _sentence_split(self, text: str) -> list[str]:
        """Split text on sentence boundaries respecting max_chunk_size."""
        sentences = self._split_into_sentences(text)
        chunks: list[str] = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If single sentence exceeds max, add it as its own chunk
            if len(sentence) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                chunks.append(sentence)
                continue

            if len(current_chunk) + len(sentence) + 1 > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = f"{current_chunk} {sentence}" if current_chunk else sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sentences if s.strip()]
