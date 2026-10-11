"""VectorStore interface + Chroma implementation. pgvector can implement the same protocol later."""
from functools import lru_cache
from typing import Protocol

from ..core.config import settings


class VectorStore(Protocol):
    def upsert(self, ids, texts, embeddings, metadatas) -> None: ...
    def delete_source(self, source_id: str) -> None: ...
    def query(self, embedding, k: int) -> list[dict]: ...
    def all_chunks(self) -> list[dict]: ...
    def count(self) -> int: ...


class ChromaStore:
    def __init__(self):
        import chromadb

        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.col = client.get_or_create_collection(
            settings.collection,
            metadata={"hnsw:space": "cosine", "embedding_model": settings.embedding_model},
        )

    def upsert(self, ids, texts, embeddings, metadatas) -> None:
        for i in range(0, len(ids), 500):  # Chroma batch limit safety
            s = slice(i, i + 500)
            self.col.upsert(
                ids=ids[s], documents=texts[s], embeddings=embeddings[s], metadatas=metadatas[s]
            )

    def delete_source(self, source_id: str) -> None:
        self.col.delete(where={"source_id": source_id})

    def query(self, embedding, k: int) -> list[dict]:
        if self.col.count() == 0:
            return []
        r = self.col.query(query_embeddings=[embedding], n_results=min(k, self.col.count()))
        return [
            {"id": i, "text": t, "metadata": m, "distance": d}
            for i, t, m, d in zip(
                r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]
            )
        ]

    def all_chunks(self) -> list[dict]:
        r = self.col.get()
        return [
            {"id": i, "text": t, "metadata": m}
            for i, t, m in zip(r["ids"], r["documents"], r["metadatas"])
        ]

    def count(self) -> int:
        return self.col.count()


@lru_cache
def get_store() -> VectorStore:
    return ChromaStore()
