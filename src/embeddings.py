"""
Text embedding generation and BM25 indexing for retrieval.

Handles embedding generation with sentence-transformers and BM25 index creation
with caching for efficient reuse.
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class TextEmbedder:
    """Generate text embeddings with sentence-transformers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the text embedder.

        Args:
            model_name: HuggingFace model name for embeddings
            cache_dir: Directory to cache the model
        """
        self.model_name = model_name
        print(f"Loading sentence-transformers model: {model_name}...")
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
        print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding

        Returns:
            Numpy array of shape (len(texts), embedding_dim)
        """
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        return embeddings

    def embed_chunks(
        self,
        chunks: list[dict],
        output_path: str,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate and save embeddings for chunks.

        Args:
            chunks: List of chunk dictionaries with 'chunk_text' field
            output_path: Path to save embeddings pickle file
            batch_size: Batch size for encoding

        Returns:
            Numpy array of embeddings
        """
        # Extract texts from chunks
        texts = [chunk['chunk_text'] for chunk in chunks]

        # Generate embeddings
        embeddings = self.embed_texts(texts, batch_size=batch_size)

        # Save embeddings
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'wb') as f:
            pickle.dump(embeddings, f)

        print(f"Embeddings saved to: {output_path}")
        print(f"Shape: {embeddings.shape}")

        return embeddings


class BM25Indexer:
    """Create BM25 index for keyword-based retrieval."""

    def __init__(self):
        """Initialize the BM25 indexer."""
        self.bm25 = None
        self.tokenized_corpus = None

    def tokenize_text(self, text: str) -> list[str]:
        """
        Tokenize text using NLTK.

        Args:
            text: Text string to tokenize

        Returns:
            List of lowercase tokens
        """
        tokens = word_tokenize(text.lower())
        return tokens

    def build_index(
        self,
        texts: list[str],
        bm25_output_path: str,
        tokens_output_path: str
    ) -> BM25Okapi:
        """
        Build BM25 index from texts.

        Args:
            texts: List of text strings to index
            bm25_output_path: Path to save BM25 index
            tokens_output_path: Path to save tokenized corpus

        Returns:
            BM25Okapi index object
        """
        print(f"Tokenizing {len(texts)} texts for BM25...")
        self.tokenized_corpus = [self.tokenize_text(text) for text in texts]

        print("Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Save BM25 index and tokenized corpus
        bm25_file = Path(bm25_output_path)
        tokens_file = Path(tokens_output_path)
        bm25_file.parent.mkdir(parents=True, exist_ok=True)
        tokens_file.parent.mkdir(parents=True, exist_ok=True)

        with open(bm25_file, 'wb') as f:
            pickle.dump(self.bm25, f)

        with open(tokens_file, 'wb') as f:
            pickle.dump(self.tokenized_corpus, f)

        print(f"BM25 index saved to: {bm25_output_path}")
        print(f"Tokenized corpus saved to: {tokens_output_path}")

        return self.bm25

    def index_chunks(
        self,
        chunks: list[dict],
        bm25_output_path: str,
        tokens_output_path: str
    ) -> BM25Okapi:
        """
        Build BM25 index from chunks.

        Args:
            chunks: List of chunk dictionaries with 'chunk_text' field
            bm25_output_path: Path to save BM25 index
            tokens_output_path: Path to save tokenized corpus

        Returns:
            BM25Okapi index object
        """
        texts = [chunk['chunk_text'] for chunk in chunks]
        return self.build_index(texts, bm25_output_path, tokens_output_path)

    @staticmethod
    def load_index(
        bm25_path: str,
        tokens_path: str
    ) -> tuple[BM25Okapi, list[list[str]]]:
        """
        Load BM25 index and tokenized corpus from files.

        Args:
            bm25_path: Path to BM25 index pickle
            tokens_path: Path to tokenized corpus pickle

        Returns:
            Tuple of (BM25Okapi index, tokenized corpus)
        """
        with open(bm25_path, 'rb') as f:
            bm25 = pickle.load(f)

        with open(tokens_path, 'rb') as f:
            tokenized_corpus = pickle.load(f)

        return bm25, tokenized_corpus


def main():
    """Test embeddings and BM25 indexing on processed chunks."""
    # Load chunks
    chunks_path = "../data/processed/chunks.json"
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks\n")

    # Generate text embeddings
    print("=== Text Embeddings ===")
    embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")
    embeddings = embedder.embed_chunks(
        chunks=chunks,
        output_path="../data/embeddings/text_embeddings.pkl",
        batch_size=32
    )

    print(f"\n=== BM25 Index ===")
    indexer = BM25Indexer()
    bm25 = indexer.index_chunks(
        chunks=chunks,
        bm25_output_path="../data/cache/bm25_index.pkl",
        tokens_output_path="../data/cache/tokenized_docs.pkl"
    )

    # Test retrieval
    print(f"\n=== Test Retrieval ===")
    query = "artificial intelligence trends"
    query_tokens = indexer.tokenize_text(query)

    # BM25 scores
    bm25_scores = bm25.get_scores(query_tokens)
    top_5_bm25 = np.argsort(bm25_scores)[::-1][:5]

    print(f"\nQuery: '{query}'")
    print(f"\nTop 5 BM25 Results:")
    for rank, idx in enumerate(top_5_bm25, 1):
        chunk = chunks[idx]
        score = bm25_scores[idx]
        print(f"{rank}. Score: {score:.3f} | {chunk['article_title'][:50]}...")
        print(f"   {chunk['chunk_text'][:100]}...\n")

    # Semantic search
    query_embedding = embedder.model.encode([query], normalize_embeddings=True)
    semantic_scores = np.dot(embeddings, query_embedding.T).flatten()
    top_5_semantic = np.argsort(semantic_scores)[::-1][:5]

    print(f"\nTop 5 Semantic Results:")
    for rank, idx in enumerate(top_5_semantic, 1):
        chunk = chunks[idx]
        score = semantic_scores[idx]
        print(f"{rank}. Score: {score:.3f} | {chunk['article_title'][:50]}...")
        print(f"   {chunk['chunk_text'][:100]}...\n")


if __name__ == "__main__":
    main()
