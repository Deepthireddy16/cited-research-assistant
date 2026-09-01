"""
retrieve_reranked.py
Two-stage retrieval:
  1. Vector search (bi-encoder) retrieves a wide candidate pool (top-20)
     — fast, casts a wide net, but only compares query/chunk independently.
  2. Cross-encoder re-ranks those 20 by scoring (query, chunk) pairs
     together — slower per-pair, but far more accurate at judging direct
     relevance. Only affordable because we've already narrowed 827 chunks
     down to 20 candidates.

Usage:
    python src/retrieve_reranked.py "How does self-attention work?"
"""

import sys
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "research_papers"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

CANDIDATE_POOL_SIZE = 20  # stage 1: cast a wide net
FINAL_TOP_K = 5           # stage 2: keep the best after re-ranking


def vector_search(query: str, model: SentenceTransformer, collection, k: int):
    query_embedding = model.encode(
        [QUERY_INSTRUCTION + query], normalize_embeddings=True
    ).tolist()

    results = collection.query(query_embeddings=query_embedding, n_results=k)

    # Reshape into a list of dicts — easier to sort/re-rank than Chroma's
    # parallel-list format.
    candidates = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        candidates.append({"text": doc, "metadata": meta, "vector_distance": dist})
    return candidates


def rerank(query: str, candidates: list, reranker: CrossEncoder, top_k: int):
    # Cross-encoder scores each (query, chunk) pair jointly — this is
    # what lets it catch relevance that independent embeddings miss.
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_k]


def print_results(results, label: str):
    print(f"\n=== {label} ===")
    for i, r in enumerate(results, start=1):
        meta = r["metadata"]
        score_str = (
            f"rerank_score: {r['rerank_score']:.4f}"
            if "rerank_score" in r
            else f"distance: {r['vector_distance']:.4f}"
        )
        print(f"\n--- Result {i} ({score_str}) ---")
        print(f"Paper: {meta['title']}")
        print(f"Page: {meta['page_number']} | arXiv: {meta['arxiv_id']}")
        print(f"Text: {r['text'][:300]}...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/retrieve_reranked.py "your question here"')
        sys.exit(1)

    query = sys.argv[1]
    print(f"Query: {query}")

    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    candidates = vector_search(query, embed_model, collection, CANDIDATE_POOL_SIZE)
    print_results(candidates[:FINAL_TOP_K], "BEFORE re-ranking (plain top-5)")

    reranked = rerank(query, candidates, reranker, FINAL_TOP_K)
    print_results(reranked, "AFTER re-ranking (top-5 of 20 candidates)")
