"""
chunk.py
Extracts text per-page from each downloaded PDF (using PyMuPDF, which
handles arXiv's 2-column layout correctly), then splits each page's
text into overlapping word-based chunks.

Output: a single JSONL file (data/chunks.jsonl) where each line is one
chunk with its text + metadata (arxiv_id, page_number, title, authors).
This is what the next stage (embedding) will read.
"""

import json
import pymupdf
from pathlib import Path

METADATA_DIR = Path("data/metadata")
CHUNKS_OUTPUT = Path("data/chunks.jsonl")

# ---- Chunking config ----
CHUNK_SIZE_WORDS = 350   # ~500 tokens
OVERLAP_WORDS = 50       # ~10-15% overlap, keeps split ideas intact
MIN_CHUNK_WORDS = 20     # skip near-empty chunks (headers, footers, blank pages)


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return a list of (page_number, page_text) for a PDF, 1-indexed."""
    doc = pymupdf.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:  # skip genuinely blank pages
            pages.append((i, text))
    doc.close()
    return pages


def chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    e.g. with 350 words/50 overlap: chunk 1 = words[0:350],
    chunk 2 = words[300:650], etc. The overlap means an idea
    split at a chunk boundary still appears whole in at least
    one chunk.
    """
    words = text.split()
    if len(words) <= CHUNK_SIZE_WORDS:
        return [text] if len(words) >= MIN_CHUNK_WORDS else []

    chunks = []
    step = CHUNK_SIZE_WORDS - OVERLAP_WORDS
    for start in range(0, len(words), step):
        chunk_words = words[start:start + CHUNK_SIZE_WORDS]
        if len(chunk_words) >= MIN_CHUNK_WORDS:
            chunks.append(" ".join(chunk_words))
        if start + CHUNK_SIZE_WORDS >= len(words):
            break
    return chunks


def process_paper(metadata: dict) -> list[dict]:
    """Extract + chunk one paper, returning a list of chunk records."""
    pdf_path = Path(metadata["local_pdf_path"])
    if not pdf_path.exists():
        print(f"    SKIPPED (PDF not found: {pdf_path})")
        return []

    records = []
    pages = extract_pages(pdf_path)

    for page_num, page_text in pages:
        page_chunks = chunk_text(page_text)
        for i, chunk in enumerate(page_chunks):
            records.append({
                "chunk_id": f"{metadata['arxiv_id']}_p{page_num}_c{i}",
                "arxiv_id": metadata["arxiv_id"],
                "title": metadata["title"],
                "authors": metadata["authors"],
                "page_number": page_num,
                "text": chunk,
            })
    return records


def main():
    metadata_files = sorted(METADATA_DIR.glob("*.json"))
    print(f"Found {len(metadata_files)} papers to process.\n")

    all_chunks = []
    for i, meta_path in enumerate(metadata_files, start=1):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print(f"[{i}/{len(metadata_files)}] {metadata['title'][:70]}...")
        chunks = process_paper(metadata)
        print(f"    -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    with open(CHUNKS_OUTPUT, "w", encoding="utf-8") as f:
        for record in all_chunks:
            f.write(json.dumps(record) + "\n")

    print(f"\nDone. {len(all_chunks)} total chunks written to {CHUNKS_OUTPUT}")


if __name__ == "__main__":
    main()
