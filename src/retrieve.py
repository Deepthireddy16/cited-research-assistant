"""
retrieve.py
Baseline retrieval: embeds a user's question (with BGE's required
query instruction prefix) and returns the top-k most similar chunks
from our Chroma vector store, with full citation metadata.

Usage:
    python src/retrieve.py "How does self-attention work?"
"""

import sys
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "research_papers"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# BGE's required instruction prefix for queries (NOT used for documents —
# see embed.py). This asymmetry is part of how BGE was trained and
# meaningfully affects retrieval quality if skipped.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

TOP_K = 5


def retrieve(query: str, k: int = TOP_K):
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    # Prefix applied here, at query time only.
    query_embedding = model.encode(
        [QUERY_INSTRUCTION + query], normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
    )

    return results


def print_results(results):
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances), start=1):
        print(f"\n--- Result {i} (distance: {dist:.4f}) ---")
        print(f"Paper: {meta['title']}")
        print(f"Authors: {meta['authors']}")
        print(f"Page: {meta['page_number']} | arXiv: {meta['arxiv_id']}")
        print(f"Text: {doc[:300]}...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/retrieve.py "your question here"')
        sys.exit(1)

    query = sys.argv[1]
    print(f"Query: {query}")
    results = retrieve(query)
    print_results(results)
