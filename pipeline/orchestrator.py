"""Pipeline orchestrator - coordinates all stages with checkpointing."""

import asyncio
import logging
import os
from dataclasses import asdict

from pipeline import PaperMetadata
from pipeline.chunker import SectionChunker
from pipeline.discovery import PaperDiscovery
from pipeline.downloader import PDFDownloader
from pipeline.embedder import BGEEmbedder
from pipeline.extractor import GROBIDExtractor
from pipeline.state import PipelineState
from pipeline.storage import QdrantStorage

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Coordinate the full pipeline with checkpointing and error handling."""

    def __init__(
        self,
        discovery: PaperDiscovery = None,
        downloader: PDFDownloader = None,
        extractor: GROBIDExtractor = None,
        chunker: SectionChunker = None,
        embedder: BGEEmbedder = None,
        storage: QdrantStorage = None,
        state: PipelineState = None,
        pdf_dir: str = "./data/pdfs",
        cleanup_pdfs: bool = True,
    ):
        self.discovery = discovery or PaperDiscovery()
        self.downloader = downloader or PDFDownloader(output_dir=pdf_dir)
        self.extractor = extractor or GROBIDExtractor()
        self.chunker = chunker or SectionChunker()
        self.embedder = embedder or BGEEmbedder()
        self.storage = storage or QdrantStorage()
        self.state = state or PipelineState()
        self.cleanup_pdfs = cleanup_pdfs

    def run(self, max_papers: int = None, batch_size: int = 500):
        """Run the full pipeline end-to-end."""
        logger.info(f"Starting pipeline (max_papers={max_papers})")

        # Ensure Qdrant collection exists
        self.storage.ensure_collection()

        # Start a run
        run_id = self.state.start_run(config={
            "max_papers": max_papers,
            "batch_size": batch_size,
        })

        total_processed = 0
        total_failed = 0

        try:
            # Phase 1: Discovery + Download + Process in streaming batches
            query = self.discovery.build_query()
            estimated = self.discovery.estimate_total(query)
            target = min(max_papers, estimated) if max_papers else estimated
            logger.info(f"Estimated {estimated} papers matching criteria, targeting {target}")

            for batch in self.discovery.iterate_batches(query=query, max_papers=max_papers):
                # Skip already-processed papers
                batch = [p for p in batch if not self.state.is_processed(p.work_id)]
                if not batch:
                    continue

                # Register discovered papers
                self.state.batch_mark_discovered([
                    (p.work_id, asdict(p)) for p in batch
                ])

                # Download PDFs
                logger.info(f"Downloading {len(batch)} PDFs...")
                download_results = asyncio.run(self.downloader.download_batch(batch))

                # Build lookup for successful downloads
                downloaded = {}
                for paper, result in zip(batch, download_results):
                    if result.success and result.pdf_path:
                        self.state.mark_downloaded(paper.work_id, result.pdf_path)
                        downloaded[paper.work_id] = (paper, result.pdf_path)
                    else:
                        self.state.mark_failed(paper.work_id, "download", result.error or "Unknown")
                        total_failed += 1

                logger.info(f"Downloaded {len(downloaded)}/{len(batch)} PDFs")

                # Process each downloaded paper
                for work_id, (paper, pdf_path) in downloaded.items():
                    try:
                        self._process_paper(paper, pdf_path)
                        total_processed += 1
                    except Exception as e:
                        logger.error(f"Failed to process {work_id}: {e}")
                        self.state.mark_failed(work_id, "processing", str(e))
                        total_failed += 1

                    # Cleanup PDF after processing
                    if self.cleanup_pdfs and os.path.exists(pdf_path):
                        os.remove(pdf_path)

                logger.info(
                    f"Batch complete: processed={total_processed}, failed={total_failed}"
                )

                # Print chunker stats periodically
                if total_processed % 100 == 0:
                    logger.info(f"Chunker stats: {self.chunker.stats}")

        finally:
            self.state.end_run(run_id, total_processed, total_failed)
            stats = self.state.get_stats()
            logger.info(f"Pipeline complete. Stats: {stats}")
            logger.info(f"Chunker final stats: {self.chunker.stats}")
            self.state.close()

    def _process_paper(self, paper: PaperMetadata, pdf_path: str):
        """Process a single paper: extract → chunk → embed → store."""
        work_id = paper.work_id

        # Extract sections via GROBID
        sections = self.extractor.extract(pdf_path)
        if not sections:
            raise ValueError(f"No sections extracted from {pdf_path}")
        self.state.mark_extracted(work_id, len(sections))

        # Chunk sections
        chunks = self.chunker.chunk_paper(work_id, sections)
        if not chunks:
            raise ValueError(f"No chunks produced for {work_id}")
        self.state.mark_chunked(work_id, len(chunks))

        # Generate embeddings
        embeddings = self.embedder.embed_chunks(chunks)

        # Store in Qdrant
        self.storage.upsert_paper(work_id, chunks, embeddings, paper)
        self.state.mark_embedded(work_id, len(chunks))

        logger.debug(f"Processed {work_id}: {len(sections)} sections, {len(chunks)} chunks")

    def resume(self, batch_size: int = 500):
        """Resume processing papers that were downloaded but not yet embedded."""
        logger.info("Resuming pipeline from checkpoint...")
        self.storage.ensure_collection()

        run_id = self.state.start_run(config={"mode": "resume"})
        total_processed = 0
        total_failed = 0

        try:
            # Process papers that are downloaded but not yet extracted/embedded
            while True:
                # Try each stage that has pending work
                for target_stage in ("downloaded", "extracted", "chunked"):
                    next_stage_idx = PipelineState.STAGES.index(target_stage) + 1
                    if next_stage_idx >= len(PipelineState.STAGES):
                        continue
                    next_stage = PipelineState.STAGES[next_stage_idx]
                    pending = self.state.get_pending(next_stage, limit=batch_size)

                    if not pending:
                        continue

                    for item in pending:
                        work_id = item["work_id"]
                        pdf_path = item.get("pdf_path")
                        metadata = item.get("metadata", {})

                        try:
                            paper = PaperMetadata(
                                work_id=work_id,
                                doi=metadata.get("doi"),
                                title=metadata.get("title", ""),
                                authors=metadata.get("authors", []),
                                year=metadata.get("year"),
                                citation_count=metadata.get("citation_count", 0),
                                pdf_url=metadata.get("pdf_url"),
                                journal=metadata.get("journal"),
                            )

                            if pdf_path and os.path.exists(pdf_path):
                                self._process_paper(paper, pdf_path)
                                total_processed += 1
                            else:
                                self.state.mark_failed(work_id, target_stage, "PDF not found")
                                total_failed += 1
                        except Exception as e:
                            logger.error(f"Resume failed for {work_id}: {e}")
                            self.state.mark_failed(work_id, target_stage, str(e))
                            total_failed += 1

                # Check if any work remains
                stats = self.state.get_stats()
                pending_stages = sum(
                    stats.get(s, 0)
                    for s in ("discovered", "downloaded", "extracted", "chunked")
                )
                if pending_stages == 0:
                    break

        finally:
            self.state.end_run(run_id, total_processed, total_failed)
            logger.info(f"Resume complete: processed={total_processed}, failed={total_failed}")
