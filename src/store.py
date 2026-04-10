from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            # TODO: initialize chromadb client + collection
            client = chromadb.Client()
            try:
                client.delete_collection(name=collection_name)
            except Exception:
                pass
            self._collection = client.get_or_create_collection(name=collection_name)

            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # TODO: build a normalized stored record for one document
        embedding = self._embedding_fn(doc.content)

        metadata = doc.metadata.copy() if doc.metadata else {}
        metadata["doc_id"] = doc.id

        record = {
            "id": f"{self._next_index}",
            "content": doc.content,
            "embedding": embedding,
            "metadata": metadata,
        }
        self._next_index += 1
        return record

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        # TODO: run in-memory similarity search over provided records
        query_embedding = self._embedding_fn(query)

        scored = []
        for r in records:
            sim = _dot(query_embedding, r["embedding"])
            scored.append((sim, r))

        scored.sort(key=lambda x: x[0], reverse=True)

        result = []
        for sim, r in scored[:top_k]:
            res = r.copy()
            res["score"] = sim
            result.append(res)
        return result

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        # TODO: embed each doc and add to store
        if self._use_chroma:
            ids = []
            documents = []
            embeddings = []
            metadatas = []

            for doc in docs:
                record = self._make_record(doc)
                ids.append(record["id"])
                documents.append(record["content"])
                embeddings.append(record["embedding"])
                metadatas.append(record["metadata"] if record["metadata"] else None)

            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas if any(metadatas) else None,
            )
        else:
            for doc in docs:
                record = self._make_record(doc)
                self._store.append(record)
        # raise NotImplementedError("Implement EmbeddingStore.add_documents")

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        # TODO: embed query, compute similarities, return top_k
        if self._use_chroma:
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            output = []
            for i in range(len(results["ids"][0])):
                output.append(
                    {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": 1.0 - results["distances"][0][i]
                    }
                )
            return output
        else:
            return self._search_records(query, self._store, top_k)
        # raise NotImplementedError("Implement EmbeddingStore.search")

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        # TODO
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)
        # raise NotImplementedError("Implement EmbeddingStore.get_collection_size")

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        # TODO: filter by metadata, then search among filtered chunks
        if self._use_chroma:
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=metadata_filter if metadata_filter else None,
                include=["documents", "metadatas", "distances"]
            )

            output = []
            for i in range(len(results["ids"][0])):
                output.append(
                    {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": 1.0 - results["distances"][0][i]
                    }
                )
            return output
        else:
            if metadata_filter:
                filtered = []
                for r in self._store:
                    match = True
                    for k, v in metadata_filter.items():
                        if r["metadata"].get(k) != v:
                            match = False
                            break
                    if match:
                        filtered.append(r)
            else:
                filtered = self._store

            return self._search_records(query, filtered, top_k)
        # raise NotImplementedError("Implement EmbeddingStore.search_with_filter")

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        # TODO: remove all stored chunks where metadata['doc_id'] == doc_id
        if self._use_chroma:
            try:
                results = self._collection.get(where={"doc_id": doc_id})
                if results and results.get("ids"):
                    self._collection.delete(where={"doc_id": doc_id})
                    return True
                return False
            except Exception:
                return False
        else:
            original_len = len(self._store)

            self._store = [
                r for r in self._store if r.get("metadata", {}).get("doc_id") != doc_id
            ]

            return len(self._store) < original_len