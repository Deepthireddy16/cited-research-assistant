# Cited Research Assistant

A retrieval-augmented generation (RAG) system that answers questions about
transformer attention mechanisms using a corpus of arXiv papers, citing its
sources with verifiable page-level references — built entirely from free,
local tools (no paid APIs).

## What it does

Given a question, the system retrieves relevant passages from a corpus of
19 arXiv papers, generates an answer grounded strictly in those passages,
and attaches citations that map back to the real paper title, authors, and
page number. If the corpus doesn't contain enough information to answer,
the system says so explicitly rather than falling back on the underlying
LLM's own (uncited) knowledge.

## Architecture

```
arXiv API
   |
   v
[ingest.py]      -> downloads PDFs + metadata
   |
   v
[chunk.py]       -> extracts per-page text (PyMuPDF), splits into
   |                overlapping chunks (350 words / 50-word overlap)
   v
[embed.py]       -> embeds chunks with BAAI/bge-small-en-v1.5,
   |                stores in a persistent Chroma vector DB
   v
[retrieval.py]   -> hybrid search: vector similarity + BM25 keyword
   |                search, merged via Reciprocal Rank Fusion
   v
[generate.py]    -> retrieves top-5 chunks, prompts a local Ollama
                     model (llama3.1:8b) with numbered sources,
                     maps citations back to verified metadata
```

## Tech stack (all free, all local)

| Component | Choice | Why |
|---|---|---|
| Corpus | arXiv API | Free, structured, includes metadata alongside PDFs |
| PDF extraction | PyMuPDF | Handles arXiv's 2-column layout correctly; preserves page numbers |
| Embedding model | BAAI/bge-small-en-v1.5 | Free, local, stronger retrieval benchmark performance than the more common MiniLM baseline |
| Vector DB | Chroma | Embedded (no server), metadata-native, zero infrastructure |
| Keyword search | rank_bm25 | Pure Python, no server, complements semantic search |
| LLM | Ollama (llama3.1:8b) | Fully local — no API key, no rate limits, no cost |

## Key design decisions

**Per-page citation granularity.** Chosen as the sweet spot between
verifiability (a reader can go check the exact page) and implementation
cost (free from PyMuPDF's page-by-page extraction).

**Chunking within page boundaries, not across them.** Keeps every chunk's
page citation unambiguous, at the small cost of occasionally splitting a
paragraph that crosses a page break.

**Hybrid retrieval (vector + BM25), not plain vector search alone.**
Vector search alone missed content with PDF-mangled math notation (e.g.
`softmax QKT √d`) that doesn't read as natural language. BM25 keyword
matching complements this blind spot. A cross-encoder re-ranker was also
tested but discarded after evaluation showed it — trained on web-search
text — performed worse on this scientific-notation-heavy corpus. See
Evaluation Journey below for the full comparison.

**Numbered source citation, not model-generated titles/IDs.** The LLM
cites `[Source N]`; the actual title, author, page, and arXiv ID are
supplied afterward from our own verified metadata — the model never has
to reproduce a precise citation string itself, which avoids hallucinated
citation details.

**Explicit grounding + refusal instruction in the system prompt.** The
model is instructed to answer only from provided sources and to say so
plainly if the sources are insufficient — this is the mechanism that
makes the system meaningfully "cited" rather than an LLM with retrieved
text pasted in front of it.

## Evaluation journey (retrieval strategy)

Three retrieval strategies were built and compared on the same test
query before settling on the final approach:

1. **Plain vector top-k** — simple baseline; missed a chunk containing
   the literal self-attention formula, likely due to PDF-mangled math
   notation not embedding well semantically.
2. **+ Cross-encoder re-ranking** — did not recover the missed formula
   chunk, and did not fix duplicate-paper crowding (re-ranking optimizes
   relevance, not diversity — a distinction worth being precise about).
   Likely limited by domain mismatch: the re-ranker was trained on
   general web search text, not scientific notation.
3. **Hybrid (vector + BM25 via Reciprocal Rank Fusion)** — the retrieval
   strategy actually used. BM25 targets exact-term matches that
   semantic search can miss. Still exposed one edge case: naive
   whitespace tokenization is brittle against PDF hyphenation artifacts
   (e.g. "self-attention" vs. "self attention").

See `eval_results.md` for full generation-stage evaluation, including a
deliberate out-of-corpus test question ("What is the capital of
France?") to verify refusal behavior — the system correctly declined to
answer rather than hallucinating, even when the retriever still returned
5 (irrelevant) candidate chunks.

## Known limitations

- **Duplicate-paper crowding:** retrieval sometimes returns multiple
  chunks from the same paper, reducing source diversity in an answer.
  MMR (Maximal Marginal Relevance) would be the natural next step to
  address this specifically.
- **BM25 tokenization brittleness:** naive whitespace/lowercase
  tokenization can miss matches across hyphenation inconsistencies
  introduced by PDF text extraction.
- **Citation formatting inconsistency:** the LLM doesn't always format
  citations identically (e.g. "Page 16" vs "p. 3") since the prompt
  doesn't enforce an exact string format.
- **Corpus scale:** currently 19 papers on one focused topic. The
  pipeline is designed to scale (ingestion is parameterized by
  `MAX_RESULTS`), but hasn't been tested at larger scale.

## Setup

1. Install [Ollama](https://ollama.com) and pull a model: `ollama pull llama3.1:8b`
2. Create a virtual environment and install dependencies:
   ```
   pip install arxiv pymupdf chromadb ollama sentence-transformers rank_bm25
   ```
3. Run the pipeline in order:
   ```
   python src/ingest.py      # fetch papers
   python src/chunk.py       # extract + chunk
   python src/embed.py       # embed + index
   python src/generate.py "your question here"
   ```
4. Optional: run the evaluation suite: `python src/eval.py`

## Possible future work

- Add MMR to address duplicate-paper crowding
- Try a scientific-domain-tuned re-ranker instead of a general web-search one
- Scale corpus size and re-run evaluation
- Improve BM25 tokenization (hyphenation normalization)
