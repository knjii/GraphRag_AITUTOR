"""Выгрузка плотных векторов для источника пар dense (эталон v2)."""

from __future__ import annotations

from types import SimpleNamespace

from rag_textbook.config import VectorStoreSettings
from rag_textbook.stores.vector_store import DENSE_VECTOR, QdrantVectorStore


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.pages = [
            ([SimpleNamespace(payload={"chunk_id": "d:00000"}, vector={DENSE_VECTOR: [1, 0]}),
              SimpleNamespace(payload={}, vector={DENSE_VECTOR: [0, 1]})], "next"),
            ([SimpleNamespace(payload={"chunk_id": "d:00001"}, vector={DENSE_VECTOR: [0.5, 0.5]})], None),
        ]

    def scroll(self, **kwargs):
        self.calls.append(kwargs)
        return self.pages[len(self.calls) - 1]


def test_qdrant_vectors_are_paged_and_keyed_by_chunk_id() -> None:
    store = QdrantVectorStore(VectorStoreSettings())
    fake = _FakeClient()
    store._client = fake
    vectors = dict(store.iter_vectors())
    assert vectors == {"d:00000": [1.0, 0.0], "d:00001": [0.5, 0.5]}
    assert fake.calls[0]["with_vectors"] == [DENSE_VECTOR]
    assert fake.calls[1]["offset"] == "next"
