"""Async PDF downloader with per-domain rate limiting."""

import asyncio
import hashlib
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import aiohttp

from pipeline import DownloadResult, PaperMetadata

logger = logging.getLogger(__name__)


class PDFDownloader:
    """Async PDF downloader with per-domain rate limiting."""

    def __init__(
        self,
        output_dir: str = "./data/pdfs",
        max_concurrent: int = 10,
        rate_limit_per_domain: float = 1.0,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        self.rate_limit_per_domain = rate_limit_per_domain
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self._domain_locks: dict[str, asyncio.Lock] = {}
        self._domain_last_request: dict[str, float] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def download_batch(self, papers: list[PaperMetadata]) -> list[DownloadResult]:
        """Download PDFs for a batch of papers."""
        connector = aiohttp.TCPConnector(limit=self.max_concurrent, ssl=False)
        async with aiohttp.ClientSession(
            timeout=self.timeout,
            connector=connector,
            headers={"User-Agent": "SciWeave-Pipeline/1.0 (research; mailto:team@desci.com)"},
        ) as session:
            tasks = [self._download_single(session, paper) for paper in papers]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        for paper, result in zip(papers, results):
            if isinstance(result, Exception):
                processed.append(DownloadResult(
                    work_id=paper.work_id,
                    success=False,
                    error=str(result),
                ))
            else:
                processed.append(result)
        return processed

    async def _download_single(
        self, session: aiohttp.ClientSession, paper: PaperMetadata
    ) -> DownloadResult:
        """Download single PDF with retries and rate limiting."""
        if not paper.pdf_url:
            return DownloadResult(work_id=paper.work_id, success=False, error="No PDF URL")

        # Determine output path using work_id hash for flat directory
        pdf_filename = f"{paper.work_id}.pdf"
        pdf_path = self.output_dir / pdf_filename

        # Skip if already downloaded
        if pdf_path.exists() and pdf_path.stat().st_size > 1000:
            return DownloadResult(work_id=paper.work_id, success=True, pdf_path=str(pdf_path))

        domain = urlparse(paper.pdf_url).netloc

        for attempt in range(self.max_retries):
            try:
                # Per-domain rate limiting
                await self._rate_limit(domain)

                async with self._semaphore:
                    async with session.get(paper.pdf_url, allow_redirects=True) as resp:
                        if resp.status != 200:
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return DownloadResult(
                                work_id=paper.work_id,
                                success=False,
                                error=f"HTTP {resp.status}",
                            )

                        content_type = resp.headers.get("content-type", "")
                        if "pdf" not in content_type and "octet-stream" not in content_type:
                            return DownloadResult(
                                work_id=paper.work_id,
                                success=False,
                                error=f"Not a PDF: {content_type}",
                            )

                        data = await resp.read()
                        if len(data) < 1000:
                            return DownloadResult(
                                work_id=paper.work_id,
                                success=False,
                                error=f"File too small: {len(data)} bytes",
                            )

                        with open(pdf_path, "wb") as f:
                            f.write(data)

                        return DownloadResult(
                            work_id=paper.work_id,
                            success=True,
                            pdf_path=str(pdf_path),
                        )

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return DownloadResult(
                    work_id=paper.work_id,
                    success=False,
                    error=f"{type(e).__name__}: {e}",
                )

        return DownloadResult(work_id=paper.work_id, success=False, error="Max retries exceeded")

    async def _rate_limit(self, domain: str):
        """Enforce per-domain rate limiting."""
        if domain not in self._domain_locks:
            self._domain_locks[domain] = asyncio.Lock()

        async with self._domain_locks[domain]:
            now = asyncio.get_event_loop().time()
            last = self._domain_last_request.get(domain, 0)
            wait = self.rate_limit_per_domain - (now - last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._domain_last_request[domain] = asyncio.get_event_loop().time()
