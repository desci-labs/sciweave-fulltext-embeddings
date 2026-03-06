"""SciWeave Full-Text Embeddings Pipeline - shared models and types."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PaperMetadata:
    """Metadata for a discovered paper."""
    work_id: str
    doi: Optional[str] = None
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    citation_count: int = 0
    pdf_url: Optional[str] = None
    journal: Optional[str] = None


@dataclass
class Section:
    """A structured section extracted from a paper."""
    title: str
    content: str
    section_type: str = "body_section"
    order: int = 0


@dataclass
class Chunk:
    """A text chunk ready for embedding."""
    paper_id: str
    text: str
    section: str
    section_type: str
    section_order: int
    chunk_index: int
    total_chunks: int = 0
    chunking_method: str = "sentence_split"


@dataclass
class DownloadResult:
    """Result of a PDF download attempt."""
    work_id: str
    success: bool
    pdf_path: Optional[str] = None
    error: Optional[str] = None
