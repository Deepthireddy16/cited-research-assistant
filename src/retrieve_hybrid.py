"""
retrieve_hybrid.py
Hybrid retrieval: runs vector search and BM25 keyword search
independently, then merges their rankings with Reciprocal Rank
Fusion (RRF).

Why: vector search misses exact-term matches in text that doesn't
look like natural language (e.g. PDF-mangled math notation) — we
saw this directly when the formula chunk for "self-attention" got
dropped by both plain vector search and the cross-encoder reranker.
BM25 finds exact term matches regardless of how "natural" the
surrounding text reads, so combining both covers each other's blind
spots.

Why RRF specifically: vector search gives cosine distances, BM25
gives an unbounded keyword-match score — different scales that
can't be meaningfully averaged. RRF sidesteps this by using each
result's RANK POSITION only, not its raw score.

Usage:
    python src/retrieve_hybrid.py "How does self-attention work?"
"""

import sys
import json
import chromadb
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("data/chunks.jsonl")
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "research_papers"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

CANDIDATE_POOL_SIZE = 20  # how many each method contributes before fusion
FINAL_TOP_K = 5
RRF_K = 60  # standard RRF constant — dampens the impact of rank 1 vs rank 2
            # being "huge" while still rewarding top ranks meaningfully


def load_chunks():
    """Load all chunks into memory, keyed by chunk_id, for BM25 + lookups."""
    chunks_by_id = {}
    corpus_ids = []
    corpus_texts = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            chunks_by_id[record["chunk_id"]] = record
            corpus_ids.append(record["chunk_id"])
            corpus_texts.append(record["text"])
    return chunks_by_id, corpus_ids, corpus_texts


def build_bm25_index(corpus_texts):
    # Simple whitespace/lowercase tokenization — good enough for BM25,
    # which cares about term overlap, not linguistic nuance.
    tokenized = [text.lower().split() for text in corpus_texts]
    return BM25Okapi(tokenized)


def vector_search_ids(query, model, collection, k):
    query_embedding = model.encode(
        [QUERY_INSTRUCTION + query], normalize_embeddings=True
    ).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)
    return results["ids"][0]  # ranked list of chunk_ids, best first


def bm25_search_ids(query, bm25, corpus_ids, k):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    # Rank corpus_ids by score, descending
    ranked = sorted(zip(corpus_ids, scores), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, _ in ranked[:k]]


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = RRF_K):
    """
    ranked_lists: a list of ranked chunk_id lists (one per retrieval method).
    Returns chunk_ids sorted by combined RRF score, best first.
    """
    scores = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)

    return sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True), scores


def print_results(chunk_ids, chunks_by_id, scores, label):
    print(f"\n=== {label} ===")
    for i, chunk_id in enumerate(chunk_ids[:FINAL_TOP_K], start=1):
        record = chunks_by_id[chunk_id]
        print(f"\n--- Result {i} (RRF score: {scores[chunk_id]:.5f}) ---")
        print(f"Paper: {record['title']}")
        print(f"Page: {record['page_number']} | arXiv: {record['arxiv_id']}")
        print(f"Text: {record['text'][:300]}...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/retrieve_hybrid.py "your question here"')
        sys.exit(1)

    query = sys.argv[1]
    print(f"Query: {query}")

    print("Loading chunks and building BM25 index...")
    chunks_by_id, corpus_ids, corpus_texts = load_chunks()
    bm25 = build_bm25_index(corpus_texts)

    print("Loading embedding model and Chroma collection...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    vector_ids = vector_search_ids(query, embed_model, collection, CANDIDATE_POOL_SIZE)
    bm25_ids = bm25_search_ids(query, bm25, corpus_ids, CANDIDATE_POOL_SIZE)

    fused_ids, scores = reciprocal_rank_fusion([vector_ids, bm25_ids])

    print_results(fused_ids, chunks_by_id, scores, "HYBRID (vector + BM25, RRF-fused)")
