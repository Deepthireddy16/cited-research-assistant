"""
ingest.py
Fetches papers from arXiv on a given topic, downloads the PDFs,
and saves structured metadata for each paper.

Why we save metadata separately (as JSON) rather than only relying
on the PDF filename: later pipeline stages (chunking, embedding,
citation display) need title/authors/page info without re-parsing
the PDF every time. This file is the "source of truth" we join
against later.
"""

import arxiv
import json
import time
import requests
from pathlib import Path

# ---- Config ----
SEARCH_QUERY = "transformer attention mechanisms"
MAX_RESULTS = 20
PDF_DIR = Path("data/pdfs")
METADATA_DIR = Path("data/metadata")

# Be polite to arXiv's free API — avoid hammering it with rapid requests
REQUEST_DELAY_SECONDS = 3


def fetch_papers():
    """Query the arXiv API and return a list of Result objects."""
    # page_size matched to MAX_RESULTS: avoids requesting a full 100-result
    # page when we only need 20 — the previous default caused an oversized
    # first request that triggered a 429 (rate limit).
    # delay_seconds/num_retries: if we still get rate-limited, back off and
    # retry instead of crashing — the arxiv library handles this internally.
    client = arxiv.Client(
        page_size=MAX_RESULTS,
        delay_seconds=5,
        num_retries=5,
    )
    search = arxiv.Search(
        query=SEARCH_QUERY,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return list(client.results(search))


def download_pdf(pdf_url: str, dest_path: Path):
    """
    Download a PDF directly via HTTP rather than relying on the arxiv
    library's built-in download method — that method's API has changed
    across versions, but a raw GET request to pdf_url is stable and
    works regardless of which arxiv library version is installed.
    """
    response = requests.get(pdf_url, timeout=30)
    response.raise_for_status()  # fail loudly if the download didn't succeed
    with open(dest_path, "wb") as f:
        f.write(response.content)


def save_metadata(paper, pdf_path: Path) -> dict:
    """Build and save a metadata dict for one paper, keyed by arxiv_id."""
    arxiv_id = paper.get_short_id()  # e.g. "1706.03762v5"

    metadata = {
        "arxiv_id": arxiv_id,
        "title": paper.title.strip(),
        "authors": [a.name for a in paper.authors],
        "abstract": paper.summary.strip(),
        "published": paper.published.isoformat(),
        "pdf_url": paper.pdf_url,
        "local_pdf_path": str(pdf_path),
    }

    meta_path = METADATA_DIR / f"{arxiv_id}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Searching arXiv for: '{SEARCH_QUERY}' (max {MAX_RESULTS} papers)...")
    papers = fetch_papers()
    print(f"Found {len(papers)} papers.\n")

    succeeded = 0
    failed = []

    for i, paper in enumerate(papers, start=1):
        arxiv_id = paper.get_short_id()
        pdf_filename = f"{arxiv_id.replace('/', '_')}.pdf"
        pdf_path = PDF_DIR / pdf_filename

        print(f"[{i}/{len(papers)}] {paper.title[:70]}...")

        try:
            # Download the PDF directly to our data/pdfs folder
            download_pdf(paper.pdf_url, pdf_path)

            # Save its metadata alongside
            save_metadata(paper, pdf_path)
            succeeded += 1
        except Exception as e:
            # One bad paper (broken link, withdrawn version, network hiccup)
            # shouldn't take down the whole batch — log it and move on.
            print(f"    SKIPPED ({e})")
            failed.append((arxiv_id, str(e)))

        # Small delay to stay within arXiv's polite-use guidelines
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nDone. {succeeded} succeeded, {len(failed)} failed.")
    if failed:
        print("Failed papers:")
        for arxiv_id, reason in failed:
            print(f"  - {arxiv_id}: {reason}")
    print("PDFs in data/pdfs/, metadata in data/metadata/.")


if __name__ == "__main__":
    main()
