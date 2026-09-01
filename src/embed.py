"""
embed.py
Embeds every chunk from data/chunks.jsonl using BAAI/bge-small-en-v1.5,
and stores the resulting vectors + metadata in a persistent Chroma
vector database (saved to disk in ./chroma_db).

Note on BGE's asymmetric convention: documents (our chunks) are
embedded plain, with no instruction prefix. Only queries get the
"Represent this sentence for searching relevant passages: " prefix,
at retrieval time — not here.
"""

import json
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("data/chunks.jsonl")
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "research_papers"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Batch size for embedding — trades memory use for speed.
# 32 is a safe default that works fine on CPU or modest GPUs.
BATCH_SIZE = 32


def load_chunks() -> list[dict]:
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def main():
    print(f"Loading embedding model: {EMBEDDING_MODEL} (first run downloads it)...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading chunks...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.\n")

    # Persistent client: writes the vector DB to disk in CHROMA_DIR,
    # so we don't have to re-embed everything every time we run a query.
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # get_or_create so re-running this script doesn't error on a second run
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    # Chroma metadata must be flat (no nested lists/dicts), so we join
    # the authors list into a single string here.
    metadatas = [
        {
            "arxiv_id": c["arxiv_id"],
            "title": c["title"],
            "authors": ", ".join(c["authors"]),
            "page_number": c["page_number"],
        }
        for c in chunks
    ]

    print("Embedding and storing chunks in batches...")
    for start in range(0, len(texts), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_texts = texts[start:end]

        # No instruction prefix here — BGE convention: documents are
        # embedded plain, only queries get the search instruction.
        embeddings = model.encode(batch_texts, normalize_embeddings=True).tolist()

        collection.add(
            ids=ids[start:end],
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=metadatas[start:end],
        )
        print(f"  {min(end, len(texts))}/{len(texts)} chunks embedded")

    print(f"\nDone. {collection.count()} chunks stored in Chroma at {CHROMA_DIR}")


if __name__ == "__main__":
    main()
