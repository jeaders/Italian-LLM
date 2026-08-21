from typing import List, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, embedding_model: str, db_path: str):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="italian_docs",
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: List[str], metadata: List[dict] = None):
        embeddings = self.embedding_model.encode(documents, show_progress_bar=False).tolist()
        ids = [f"doc_{i}" for i in range(len(documents))]
        metadatas = metadata or [{} for _ in documents]
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        logger.info(f"Added {len(documents)} documents to vector DB")

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)
        return [
            {
                "text": doc,
                "score": float(score),
                "metadata": meta,
            }
            for doc, score, meta in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            )
        ]

    def get_context(self, query: str, top_k: int = 3) -> str:
        results = self.search(query, top_k)
        return "\n\n".join([r["text"] for r in results])

    def delete_collection(self):
        self.client.delete_collection("italian_docs")

    def count(self) -> int:
        return self.collection.count()
