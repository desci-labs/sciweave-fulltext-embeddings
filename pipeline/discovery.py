"""Discover OA papers from Elasticsearch matching V0 criteria."""

import logging
import os
from typing import Iterator

from elasticsearch import Elasticsearch, helpers

from pipeline import PaperMetadata

logger = logging.getLogger(__name__)


class PaperDiscovery:
    """Query Elasticsearch for OA papers matching V0 criteria."""

    def __init__(
        self,
        es_url: str = None,
        es_user: str = None,
        es_password: str = None,
        index: str = None,
        batch_size: int = 500,
    ):
        self.es_url = es_url or os.getenv("ES_HOST")
        self.es_user = es_user or os.getenv("ES_USER")
        self.es_password = es_password or os.getenv("ES_PWD")
        self.index = index or os.getenv("ES_INDEX", "works_opt")
        self.batch_size = batch_size

        self.client = Elasticsearch(
            self.es_url,
            basic_auth=(self.es_user, self.es_password),
            verify_certs=False,
            request_timeout=60,
        )

    def build_query(
        self,
        min_year: int = 2015,
        min_citations: int = 10,
        language: str = "en",
    ) -> dict:
        """Build ES query for V0 criteria: OA, English, 2015+, >10 citations."""
        return {
            "bool": {
                "must": [
                    {"range": {"publication_year": {"gte": min_year}}},
                    {"range": {"cited_by_count": {"gte": min_citations}}},
                    {"term": {"language": language}},
                ],
                "should": [
                    {"nested": {"path": "best_locations", "query": {"exists": {"field": "best_locations.pdf_url"}}}},
                    {"nested": {"path": "locations", "query": {"exists": {"field": "locations.pdf_url"}}}},
                ],
                "minimum_should_match": 1,
            }
        }

    def estimate_total(self, query: dict = None) -> int:
        """Count total matching papers."""
        if query is None:
            query = self.build_query()
        resp = self.client.count(index=self.index, query=query)
        return resp["count"]

    def iterate_batches(
        self,
        query: dict = None,
        max_papers: int = None,
    ) -> Iterator[list[PaperMetadata]]:
        """Scroll through results yielding batches of paper metadata.

        Results are sorted by cited_by_count desc (highest quality first).
        """
        if query is None:
            query = self.build_query()

        source_fields = [
            "work_id", "doi", "title", "authors.display_name",
            "publication_year", "cited_by_count",
            "best_locations.pdf_url", "best_locations.source_id",
            "best_locations.display_name",
            "locations.pdf_url", "locations.display_name",
        ]

        total_yielded = 0
        batch = []

        scan_iter = helpers.scan(
            client=self.client,
            query={"query": query, "sort": [{"cited_by_count": "desc"}]},
            index=self.index,
            size=self.batch_size,
            scroll="10m",
            preserve_order=True,
            _source=source_fields,
        )

        for doc in scan_iter:
            if max_papers and total_yielded >= max_papers:
                break

            source = doc["_source"]
            paper = self._parse_paper(source)
            if paper and paper.pdf_url:
                batch.append(paper)
                total_yielded += 1

            if len(batch) >= self.batch_size:
                logger.info(f"Yielding batch of {len(batch)} papers (total: {total_yielded})")
                yield batch
                batch = []

        if batch:
            logger.info(f"Yielding final batch of {len(batch)} papers (total: {total_yielded})")
            yield batch

    def _parse_paper(self, source: dict) -> PaperMetadata | None:
        """Parse ES document into PaperMetadata."""
        work_id = source.get("work_id", "")
        if "/" in work_id:
            work_id = work_id.split("/")[-1]

        if not work_id:
            return None

        # Extract PDF URL: best_locations first, then locations
        pdf_url = None
        journal = None
        for loc in source.get("best_locations", []):
            if loc.get("pdf_url"):
                pdf_url = loc["pdf_url"]
                journal = loc.get("display_name")
                break
        if not pdf_url:
            for loc in source.get("locations", []):
                if loc.get("pdf_url"):
                    pdf_url = loc["pdf_url"]
                    journal = loc.get("display_name")
                    break

        # Extract authors
        authors = [
            a.get("display_name", "")
            for a in source.get("authors", [])
            if a.get("display_name")
        ]

        doi = source.get("doi", "")
        if doi and "doi.org/" in doi:
            doi = doi.split("doi.org/")[-1]

        return PaperMetadata(
            work_id=work_id,
            doi=doi or None,
            title=source.get("title", ""),
            authors=authors,
            year=int(source["publication_year"]) if source.get("publication_year") is not None else None,
            citation_count=source.get("cited_by_count", 0),
            pdf_url=pdf_url,
            journal=journal,
        )
