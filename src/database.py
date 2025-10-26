"""
Vector database operations using ChromaDB for multimodal retrieval.

Handles storage and retrieval of text and image embeddings with metadata
filtering support.
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np
from chromadb.config import Settings


class MultimodalDB:
    """Multimodal vector database with ChromaDB."""

    def __init__(self, persist_directory: str = "../data/chroma_db"):
        """
        Initialize ChromaDB client and collections.

        Args:
            persist_directory: Directory for persistent storage
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Collections will be created/loaded on demand
        self.text_collection = None
        self.image_collection = None

    def initialize_text_collection(
        self,
        chunks_path: str,
        embeddings_path: str,
        collection_name: str = "text_chunks"
    ) -> chromadb.Collection:
        """
        Initialize text chunks collection with embeddings and metadata.

        Args:
            chunks_path: Path to chunks JSON
            embeddings_path: Path to text embeddings pickle
            collection_name: Name for the collection

        Returns:
            ChromaDB collection object
        """
        print(f"\n=== Initializing Text Collection ===")

        # Load chunks
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        # Load embeddings
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)

        print(f"Loaded {len(chunks)} chunks and {len(embeddings)} embeddings")

        # Get or create collection
        try:
            self.text_collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
            print(f"Collection count: {self.text_collection.count()}")
        except Exception:
            # Create new collection
            self.text_collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Text chunks with semantic embeddings"}
            )
            print(f"Created new collection: {collection_name}")

            # Prepare data for ChromaDB
            ids = [chunk['chunk_id'] for chunk in chunks]
            documents = [chunk['chunk_text'] for chunk in chunks]
            metadatas = [
                {
                    'article_id': chunk['article_id'],
                    'article_title': chunk['article_title'],
                    'article_url': chunk['article_url'],
                    'chunk_index': chunk['chunk_index'],
                    'total_chunks': chunk['total_chunks'],
                    'word_count': chunk['word_count']
                }
                for chunk in chunks
            ]

            # Add to collection in batches
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_docs = documents[i:i + batch_size]
                batch_meta = metadatas[i:i + batch_size]
                batch_embs = embeddings[i:i + batch_size].tolist()

                self.text_collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_meta,
                    embeddings=batch_embs
                )

            print(f"Added {len(ids)} text chunks to collection")

        return self.text_collection

    def initialize_image_collection(
        self,
        metadata_path: str,
        embeddings_path: str,
        collection_name: str = "images"
    ) -> chromadb.Collection:
        """
        Initialize images collection with embeddings and metadata.

        Args:
            metadata_path: Path to image metadata JSON
            embeddings_path: Path to image embeddings pickle
            collection_name: Name for the collection

        Returns:
            ChromaDB collection object
        """
        print(f"\n=== Initializing Image Collection ===")

        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Load embeddings
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)

        print(f"Loaded {len(metadata)} images and {len(embeddings)} embeddings")

        # Get or create collection
        try:
            self.image_collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
            print(f"Collection count: {self.image_collection.count()}")
        except Exception:
            # Create new collection
            self.image_collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Image embeddings with CLIP"}
            )
            print(f"Created new collection: {collection_name}")

            # Prepare data for ChromaDB
            ids = [f"img_{i}" for i in range(len(metadata))]
            documents = [meta['image_path'] for meta in metadata]
            metadatas = [
                {
                    'article_id': meta['article_id'],
                    'article_title': meta['article_title'],
                    'image_path': meta['image_path'],
                    'chunk_id': meta['chunk_id'],
                    'full_path': meta['full_path']
                }
                for meta in metadata
            ]

            # Add to collection
            self.image_collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings.tolist()
            )

            print(f"Added {len(ids)} images to collection")

        return self.image_collection

    def query_text(
        self,
        query_embedding,
        n_results: int = 5,
        where: Optional[dict] = None
    ) -> dict:
        """
        Query text chunks collection.

        Args:
            query_embedding: Query embedding (numpy array or list)
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            Query results from ChromaDB
        """
        if self.text_collection is None:
            raise ValueError("Text collection not initialized")

        # Convert numpy to list if needed
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()

        return self.text_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )

    def query_images(
        self,
        query_embedding,
        n_results: int = 5,
        where: Optional[dict] = None
    ) -> dict:
        """
        Query images collection.

        Args:
            query_embedding: Query embedding (numpy array or list)
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            Query results from ChromaDB
        """
        if self.image_collection is None:
            raise ValueError("Image collection not initialized")

        # Convert numpy to list if needed
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()

        return self.image_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )

    def reset_database(self):
        """Reset the database (delete all collections)."""
        self.client.reset()
        self.text_collection = None
        self.image_collection = None
        print("Database reset complete")


def main():
    """Test ChromaDB initialization and querying."""
    # Initialize database
    db = MultimodalDB(persist_directory="../data/chroma_db")

    # Initialize collections
    db.initialize_text_collection(
        chunks_path="../data/processed/chunks.json",
        embeddings_path="../data/embeddings/text_embeddings.pkl"
    )

    db.initialize_image_collection(
        metadata_path="../data/embeddings/image_metadata.json",
        embeddings_path="../data/embeddings/image_embeddings.pkl"
    )

    # Test text query with dummy embedding
    print("\n=== Test: Text Query ===")
    dummy_text_emb = np.random.rand(384)  # 384-dim for text

    results = db.query_text(dummy_text_emb, n_results=3)
    print(f"Found {len(results['ids'][0])} results")
    for i, (id, doc, meta, dist) in enumerate(zip(
        results['ids'][0],
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\n{i+1}. ID: {id}")
        print(f"   Article: {meta['article_title'][:50]}...")
        print(f"   Text: {doc[:80]}...")
        print(f"   Distance: {dist:.3f}")

    # Test metadata filtering
    print("\n=== Test: Metadata Filtering ===")
    filtered_results = db.query_text(
        dummy_text_emb,
        n_results=3,
        where={"article_id": 1}
    )
    print(f"Found {len(filtered_results['ids'][0])} results for article_id=1")
    for id, meta in zip(filtered_results['ids'][0], filtered_results['metadatas'][0]):
        print(f"  - {id} (article {meta['article_id']})")

    # Test image query
    print("\n=== Test: Image Query ===")
    dummy_image_emb = np.random.rand(512)  # 512-dim for images

    img_results = db.query_images(dummy_image_emb, n_results=3)
    print(f"Found {len(img_results['ids'][0])} image results")
    for i, (id, doc, meta, dist) in enumerate(zip(
        img_results['ids'][0],
        img_results['documents'][0],
        img_results['metadatas'][0],
        img_results['distances'][0]
    )):
        print(f"\n{i+1}. ID: {id}")
        print(f"   Image: {meta['image_path']}")
        print(f"   Article: {meta['article_title'][:50]}...")
        print(f"   Distance: {dist:.3f}")

    print("\n=== Database Statistics ===")
    print(f"Text chunks: {db.text_collection.count()}")
    print(f"Images: {db.image_collection.count()}")


if __name__ == "__main__":
    main()
