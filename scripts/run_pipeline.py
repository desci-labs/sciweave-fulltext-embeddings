#!/usr/bin/env python3
"""CLI entry point for the SciWeave full-text embeddings pipeline."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from pipeline.orchestrator import PipelineOrchestrator
from pipeline.discovery import PaperDiscovery
from pipeline.downloader import PDFDownloader
from pipeline.extractor import GROBIDExtractor
from pipeline.chunker import HybridChunker
from pipeline.embedder import BGEEmbedder
from pipeline.storage import QdrantStorage
from pipeline.state import PipelineState


def main():
    parser = argparse.ArgumentParser(
        description="SciWeave Full-Text Embeddings Pipeline"
    )
    parser.add_argument(
        "--max-papers", type=int, default=100_000,
        help="Maximum number of papers to process (default: 100000)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=500,
        help="Batch size for discovery and processing (default: 500)"
    )
    parser.add_argument(
        "--pdf-dir", type=str, default="./data/pdfs",
        help="Directory to store downloaded PDFs"
    )
    parser.add_argument(
        "--state-db", type=str, default="pipeline_state.db",
        help="Path to SQLite state database"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint instead of starting fresh"
    )
    parser.add_argument(
        "--no-cleanup", action="store_true",
        help="Keep PDFs after processing (default: delete after embedding)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--embedding-device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Device for embedding model"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reduce noise from third-party libraries
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger = logging.getLogger("pipeline")
    logger.info(f"Pipeline starting with max_papers={args.max_papers}")

    # Initialize components
    orchestrator = PipelineOrchestrator(
        discovery=PaperDiscovery(batch_size=args.batch_size),
        downloader=PDFDownloader(output_dir=args.pdf_dir),
        extractor=GROBIDExtractor(),
        chunker=HybridChunker(),
        embedder=BGEEmbedder(),
        storage=QdrantStorage(),
        state=PipelineState(db_path=args.state_db),
        pdf_dir=args.pdf_dir,
        cleanup_pdfs=not args.no_cleanup,
    )

    if args.resume:
        orchestrator.resume(batch_size=args.batch_size)
    else:
        orchestrator.run(max_papers=args.max_papers, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
