"""
Vector database with Qdrant for multimodal retrieval.

Handles text and image embeddings with advanced filtering.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
    Range,
    VectorParams,
)


class MultimodalDB:
    """Multimodal vector database with Qdrant."""

    def __init__(self, persist_directory: str = "../data/qdrant_db"):
        """
        Initialize Qdrant client.

        Args:
            persist_directory: Directory for persistent storage
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.persist_directory))
        self.text_collection = "text_chunks"
        self.image_collection = "images"

    def initialize_text_collection(self, chunks_path: str, embeddings_path: str):
        """Initialize text collection with chunks and embeddings."""
        print(f"\n=== Initializing Text Collection ===")

        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)

        print(f"Loaded {len(chunks)} chunks and {len(embeddings)} embeddings")

        if self.client.collection_exists(self.text_collection):
            self.client.delete_collection(self.text_collection)

        self.client.create_collection(
            collection_name=self.text_collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

        points = [
            PointStruct(
                id=idx,
                vector=emb.tolist(),
                payload={
                    'chunk_id': chunk['chunk_id'],
                    'article_id': chunk['article_id'],
                    'article_title': chunk['article_title'],
                    'article_url': chunk['article_url'],
                    'article_date': chunk.get('article_date', ''),
                    'article_timestamp': chunk.get('article_timestamp'),
                    'article_categories': chunk.get('article_categories', []),
                    'chunk_index': chunk['chunk_index'],
                    'total_chunks': chunk['total_chunks'],
                    'word_count': chunk['word_count'],
                    'chunk_text': chunk['chunk_text']
                }
            )
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]

        for i in range(0, len(points), 100):
            self.client.upsert(collection_name=self.text_collection, points=points[i:i+100])

        print(f"Added {len(points)} text chunks")

    def initialize_image_collection(self, metadata_path: str, embeddings_path: str):
        """Initialize image collection with metadata and embeddings."""
        print(f"\n=== Initializing Image Collection ===")

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)

        print(f"Loaded {len(metadata)} images and {len(embeddings)} embeddings")

        if self.client.collection_exists(self.image_collection):
            self.client.delete_collection(self.image_collection)

        self.client.create_collection(
            collection_name=self.image_collection,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )

        points = [
            PointStruct(
                id=idx,
                vector=emb.tolist(),
                payload={
                    'article_id': meta['article_id'],
                    'article_title': meta['article_title'],
                    'article_url': meta.get('article_url', ''),
                    'article_date': meta.get('article_date', ''),
                    'article_timestamp': meta.get('article_timestamp'),
                    'article_categories': meta.get('article_categories', []),
                    'image_path': meta['image_path'],
                    'full_path': meta['full_path']
                }
            )
            for idx, (meta, emb) in enumerate(zip(metadata, embeddings))
        ]

        self.client.upsert(collection_name=self.image_collection, points=points)
        print(f"Added {len(points)} images")

    def query_text(self, query_embedding, n_results: int = 5, query_filter: dict | None = None) -> list:
        """Query text chunks."""
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        return self.client.search(
            collection_name=self.text_collection,
            query_vector=query_embedding,
            limit=n_results,
            query_filter=query_filter
        )

    def query_images(self, query_embedding, n_results: int = 5, query_filter: dict | None = None) -> list:
        """Query images."""
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        return self.client.search(
            collection_name=self.image_collection,
            query_vector=query_embedding,
            limit=n_results,
            query_filter=query_filter
        )


def main():
    """Test Qdrant initialization and filtering."""
    db = MultimodalDB(persist_directory="../data/qdrant_db")

    db.initialize_text_collection(
        chunks_path="../data/processed/chunks.json",
        embeddings_path="../data/embeddings/text_embeddings.pkl"
    )

    db.initialize_image_collection(
        metadata_path="../data/embeddings/image_metadata.json",
        embeddings_path="../data/embeddings/image_embeddings.pkl"
    )

    dummy_text_emb = np.random.rand(384)
    dummy_img_emb = np.random.rand(512)

    # Test 1: Basic query
    print("\n=== Test 1: Basic Text Query ===")
    results = db.query_text(dummy_text_emb, n_results=3)
    for i, r in enumerate(results):
        print(f"{i+1}. {r.payload['article_title'][:50]}... | Score: {r.score:.3f}")

    # Test 2: Multi-category filter
    print("\n=== Test 2: Multi-Category Filter ===")
    results = db.query_text(
        dummy_text_emb,
        n_results=3,
        query_filter=Filter(must=[FieldCondition(key="article_categories", match=MatchAny(any=["ML Research", "Business"]))])
    )
    for r in results:
        print(f"  - {r.payload['article_title'][:50]} | {r.payload['article_categories']}")

    # Test 3: Date range filter (October 2025 timestamps)
    print("\n=== Test 3: Date Range (October 2025) ===")
    oct_start = int(datetime(2025, 10, 1).timestamp())
    nov_start = int(datetime(2025, 11, 1).timestamp())
    results = db.query_text(
        dummy_text_emb,
        n_results=5,
        query_filter=Filter(must=[FieldCondition(key="article_timestamp", range=Range(gte=oct_start, lt=nov_start))])
    )
    for r in results:
        print(f"  - {r.payload['article_title'][:40]} | {r.payload['article_date'][:10]}")

    # Test 4: Image query
    print("\n=== Test 4: Image Query ===")
    results = db.query_images(dummy_img_emb, n_results=3)
    for r in results:
        print(f"  - {r.payload['image_path']} | {r.payload['article_title'][:40]}")

    # Stats
    print("\n=== Statistics ===")
    print(f"Text chunks: {db.client.get_collection(db.text_collection).points_count}")
    print(f"Images: {db.client.get_collection(db.image_collection).points_count}")


if __name__ == "__main__":
    main()
