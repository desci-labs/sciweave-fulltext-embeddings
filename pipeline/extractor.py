"""Extract structured text from PDFs using GROBID."""

import logging
import os
import re
import xml.etree.ElementTree as ET

import requests

from pipeline import Section

logger = logging.getLogger(__name__)

TEI_NS = "{http://www.tei-c.org/ns/1.0}"


class GROBIDExtractor:
    """Extract structured text from PDFs using a self-hosted GROBID instance."""

    def __init__(self, grobid_url: str = None, timeout: int = 60):
        self.grobid_url = grobid_url or os.getenv("GROBID_URL", "http://localhost:8070")
        self.timeout = timeout

    def extract(self, pdf_path: str) -> dict[str, Section]:
        """Send PDF to GROBID, parse TEI XML into structured sections."""
        tei_xml = self._call_grobid(pdf_path)
        if not tei_xml:
            return {}
        return self._parse_tei(tei_xml)

    def _call_grobid(self, pdf_path: str) -> str | None:
        """Send PDF to GROBID and return TEI XML response."""
        url = f"{self.grobid_url}/api/processFulltextDocument"
        try:
            with open(pdf_path, "rb") as f:
                resp = requests.post(
                    url,
                    files={"input": (os.path.basename(pdf_path), f, "application/pdf")},
                    timeout=self.timeout,
                )
            if resp.status_code == 200:
                return resp.text
            logger.warning(f"GROBID returned {resp.status_code} for {pdf_path}")
            return None
        except requests.RequestException as e:
            logger.error(f"GROBID request failed for {pdf_path}: {e}")
            return None

    def _parse_tei(self, tei_xml: str) -> dict[str, Section]:
        """Parse TEI XML into a dict of sections."""
        try:
            root = ET.fromstring(tei_xml)
        except ET.ParseError as e:
            logger.error(f"Failed to parse TEI XML: {e}")
            return {}

        sections: dict[str, Section] = {}
        order = 0

        # Extract title
        title_el = root.find(f".//{TEI_NS}titleStmt/{TEI_NS}title")
        if title_el is not None and title_el.text:
            sections["title"] = Section(
                title="Title",
                content=title_el.text.strip(),
                section_type="title",
                order=order,
            )
            order += 1

        # Extract abstract
        abstract_el = root.find(f".//{TEI_NS}profileDesc/{TEI_NS}abstract")
        if abstract_el is not None:
            abstract_text = self._extract_text(abstract_el)
            if abstract_text.strip():
                sections["abstract"] = Section(
                    title="Abstract",
                    content=abstract_text.strip(),
                    section_type="abstract",
                    order=order,
                )
                order += 1

        # Extract body sections
        body = root.find(f".//{TEI_NS}body")
        if body is not None:
            for div in body.findall(f"{TEI_NS}div"):
                head = div.find(f"{TEI_NS}head")
                section_title = head.text.strip() if head is not None and head.text else f"Section {order}"

                # Get section number if present
                section_n = head.get("n", "") if head is not None else ""
                if section_n:
                    section_title = f"{section_n} {section_title}"

                content = self._extract_div_text(div)
                if not content.strip():
                    continue

                # Determine section type from title
                section_type = self._classify_section(section_title)
                section_key = re.sub(r"[^a-z0-9]+", "_", section_title.lower()).strip("_")

                sections[section_key] = Section(
                    title=section_title,
                    content=content.strip(),
                    section_type=section_type,
                    order=order,
                )
                order += 1

        # Extract references/bibliography section
        back = root.find(f".//{TEI_NS}back")
        if back is not None:
            refs = back.findall(f".//{TEI_NS}biblStruct")
            if refs:
                ref_texts = []
                for ref in refs:
                    title = ref.find(f".//{TEI_NS}title")
                    if title is not None and title.text:
                        ref_texts.append(title.text.strip())
                if ref_texts:
                    sections["references"] = Section(
                        title="References",
                        content="\n".join(ref_texts),
                        section_type="references",
                        order=order,
                    )

        return sections

    def _extract_text(self, element: ET.Element) -> str:
        """Recursively extract all text from an element."""
        texts = []
        if element.text:
            texts.append(element.text)
        for child in element:
            texts.append(self._extract_text(child))
            if child.tail:
                texts.append(child.tail)
        return " ".join(texts)

    def _extract_div_text(self, div: ET.Element) -> str:
        """Extract text from a div, skipping the head element."""
        texts = []
        for child in div:
            tag = child.tag.replace(TEI_NS, "")
            if tag == "head":
                continue
            texts.append(self._extract_text(child))
        return " ".join(texts)

    def _classify_section(self, title: str) -> str:
        """Classify section type based on title keywords."""
        title_lower = title.lower()
        if any(kw in title_lower for kw in ["abstract"]):
            return "abstract"
        if any(kw in title_lower for kw in ["introduction", "background"]):
            return "introduction"
        if any(kw in title_lower for kw in ["method", "material", "experimental", "procedure"]):
            return "methods"
        if any(kw in title_lower for kw in ["result", "finding"]):
            return "results"
        if any(kw in title_lower for kw in ["discussion"]):
            return "discussion"
        if any(kw in title_lower for kw in ["conclusion", "summary"]):
            return "conclusion"
        if any(kw in title_lower for kw in ["reference", "bibliography"]):
            return "references"
        if any(kw in title_lower for kw in ["acknowledgment", "acknowledgement"]):
            return "acknowledgments"
        if any(kw in title_lower for kw in ["appendix", "supplement"]):
            return "appendix"
        return "body_section"
