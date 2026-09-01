"""
retrieval.py
The finalized retrieval module: hybrid vector + BM25 search, fused
with Reciprocal Rank Fusion. This is what generate.py imports —
kept separate from the retrieve_*.py experiment scripts, which
remain as documentation of the evaluation process (plain vector
search -> cross-encoder rerank -> hybrid), each with a distinct,
documented failure mode found along the way.

Known limitation (documented, not yet fixed): BM25 tokenization is
naive whitespace splitting, which is brittle against PDF-extraction
artifacts like inconsistent hyphenation ("self-attention" vs.
"self attention"). A future improvement would use a more robust
tokenizer or a hyphenation-normalization step.
"""

import json
from pathlib import Path
from rank_bm25 import BM25Okapi
import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("data/chunks.jsonl")
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "research_papers"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

CANDIDATE_POOL_SIZE = 20
DEFAULT_TOP_K = 5
RRF_K = 60


class Retriever:
    """
    Loads all indexes once (embedding model, Chroma collection, BM25
    index) so repeated .retrieve() calls — e.g. across a chat session —
    don't pay that setup cost every time.
    """

    def __init__(self):
        self.chunks_by_id, self.corpus_ids, corpus_texts = self._load_chunks()
        self.bm25 = self._build_bm25_index(corpus_texts)
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = client.get_collection(name=COLLECTION_NAME)

    def _load_chunks(self):
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

    def _build_bm25_index(self, corpus_texts):
        tokenized = [text.lower().split() for text in corpus_texts]
        return BM25Okapi(tokenized)

    def _vector_search_ids(self, query, k):
        query_embedding = self.embed_model.encode(
            [QUERY_INSTRUCTION + query], normalize_embeddings=True
        ).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=k)
        return results["ids"][0]

    def _bm25_search_ids(self, query, k):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(zip(self.corpus_ids, scores), key=lambda x: x[1], reverse=True)
        return [chunk_id for chunk_id, _ in ranked[:k]]

    def _reciprocal_rank_fusion(self, ranked_lists, k=RRF_K):
        scores = {}
        for ranked_list in ranked_lists:
            for rank, chunk_id in enumerate(ranked_list, start=1):
                scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
        return sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """Returns the top_k chunk records (dicts with text + metadata)."""
        vector_ids = self._vector_search_ids(query, CANDIDATE_POOL_SIZE)
        bm25_ids = self._bm25_search_ids(query, CANDIDATE_POOL_SIZE)
        fused_ids = self._reciprocal_rank_fusion([vector_ids, bm25_ids])
        return [self.chunks_by_id[cid] for cid in fused_ids[:top_k]]
