"""Hybrid section-aware chunker: sentence splitting + LLM semantic chunking."""

import json
import logging
import os
import re

from openai import OpenAI

from pipeline import Chunk, Section

logger = logging.getLogger(__name__)

SEMANTIC_CHUNK_PROMPT = """You are a text chunking assistant for academic papers. Split the following section text into semantically coherent chunks. Each chunk should be 800-1200 characters and contain a complete thought or closely related ideas.

Rules:
- Never split mid-sentence
- Keep related concepts together (e.g. a claim and its evidence)
- Each chunk should be understandable on its own
- Return ONLY a JSON array of strings, where each string is a chunk
- Do not add any commentary, just the JSON array

Section text:
{text}"""


class HybridChunker:
    """Hybrid section-aware chunker.

    Tier 1 (short sections ≤ MAX_SECTION_CHARS): Simple sentence-boundary splitting.
    Tier 2 (long sections > MAX_SECTION_CHARS): LLM-based semantic chunking via Chutes API.
    Falls back to sentence splitting if LLM call fails.
    """

    MAX_SECTION_CHARS = 3000

    def __init__(
        self,
        max_chunk_size: int = 1000,
        chutes_api_key: str = None,
        semantic_model: str = None,
    ):
        self.max_chunk_size = max_chunk_size
        self.chutes_api_key = chutes_api_key or os.getenv("CHUTES_API_KEY")
        self.semantic_model = semantic_model or os.getenv(
            "CHUTES_FAST_MODEL", "openai/gpt-oss-120b-TEE"
        )

        # Initialize Chutes client (OpenAI-compatible API)
        if self.chutes_api_key:
            self.llm_client = OpenAI(
                api_key=self.chutes_api_key,
                base_url="https://llm.chutes.ai/v1",
            )
        else:
            self.llm_client = None
            logger.warning("No CHUTES_API_KEY set; semantic chunking disabled, using sentence split only")

        # Stats tracking
        self.stats = {"sentence_split": 0, "semantic": 0, "semantic_fallback": 0}

    def chunk_paper(self, paper_id: str, sections: dict[str, Section]) -> list[Chunk]:
        """Chunk all sections of a paper using hybrid strategy."""
        all_chunks: list[Chunk] = []

        for section_key, section in sections.items():
            if not section.content.strip():
                continue

            chunk_texts, method = self._chunk_section(section)

            for i, text in enumerate(chunk_texts):
                all_chunks.append(Chunk(
                    paper_id=paper_id,
                    text=text,
                    section=section_key,
                    section_type=section.section_type,
                    section_order=section.order,
                    chunk_index=i,
                    total_chunks=len(chunk_texts),
                    chunking_method=method,
                ))

        return all_chunks

    def _chunk_section(self, section: Section) -> tuple[list[str], str]:
        """Route to sentence splitting or semantic chunking based on length."""
        content = section.content.strip()
        if not content:
            return [], "sentence_split"

        # Short sections or no LLM available: use sentence splitting
        if len(content) <= self.MAX_SECTION_CHARS or not self.llm_client:
            chunks = self._sentence_split(content)
            self.stats["sentence_split"] += 1
            return chunks, "sentence_split"

        # Long sections: try semantic chunking
        chunks = self._semantic_chunk(content)
        if chunks:
            self.stats["semantic"] += 1
            return chunks, "semantic"

        # Fallback to sentence splitting
        logger.warning(f"Semantic chunking failed for section '{section.title}', falling back to sentence split")
        self.stats["semantic_fallback"] += 1
        chunks = self._sentence_split(content)
        return chunks, "sentence_split"

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

    def _semantic_chunk(self, text: str) -> list[str] | None:
        """Use LLM via Chutes API to semantically chunk long sections.

        Returns None on failure (caller should fall back to sentence splitting).
        """
        try:
            response = self.llm_client.chat.completions.create(
                model=self.semantic_model,
                messages=[
                    {"role": "user", "content": SEMANTIC_CHUNK_PROMPT.format(text=text)},
                ],
                temperature=0.1,
                max_tokens=4096,
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON array from response
            # Handle cases where LLM wraps in markdown code blocks
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)

            chunks = json.loads(content)
            if not isinstance(chunks, list) or not all(isinstance(c, str) for c in chunks):
                logger.warning("LLM returned non-list or non-string chunks")
                return None

            # Filter out empty chunks
            chunks = [c.strip() for c in chunks if c.strip()]
            if not chunks:
                return None

            return chunks

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM chunking response as JSON: {e}")
            return None
        except Exception as e:
            logger.warning(f"Semantic chunking LLM call failed: {e}")
            return None

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex."""
        # Split on sentence-ending punctuation followed by space or end
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sentences if s.strip()]
