"""
Multimodal retrieval system combining BM25, semantic search, and CLIP.

Implements hybrid text retrieval and multimodal fusion for comprehensive
document and image retrieval.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from database import MultimodalDB
from embeddings import BM25Indexer


class MultimodalRetriever:
    """Multimodal retrieval system with hybrid text and image search."""

    def __init__(
        self,
        db: MultimodalDB,
        text_model_name: str = "all-MiniLM-L6-v2",
        clip_model_name: str = "openai/clip-vit-base-patch32",
        bm25_path: str = "../data/cache/bm25_index.pkl",
        tokens_path: str = "../data/cache/tokenized_docs.pkl",
        alpha: float = 0.4,
        beta: float = 0.7
    ):
        """
        Initialize multimodal retriever.

        Args:
            db: MultimodalDB instance
            text_model_name: Sentence transformer model name
            clip_model_name: CLIP model name
            bm25_path: Path to BM25 index
            tokens_path: Path to tokenized corpus
            alpha: Weight for BM25 in text fusion (0-1)
            beta: Weight for text in multimodal fusion (0-1)
        """
        self.db = db
        self.alpha = alpha
        self.beta = beta

        # Load text embedding model
        print(f"Loading text model: {text_model_name}...")
        self.text_model = SentenceTransformer(text_model_name)

        # Load CLIP model
        print(f"Loading CLIP model: {clip_model_name}...")
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model.eval()

        # Load BM25 index
        print("Loading BM25 index...")
        with open(bm25_path, 'rb') as f:
            self.bm25 = pickle.load(f)
        with open(tokens_path, 'rb') as f:
            self.tokenized_corpus = pickle.load(f)

        # BM25 tokenizer
        self.bm25_indexer = BM25Indexer()

        print("Retriever initialized successfully")

    def retrieve_text(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[dict] = None,
        use_hybrid: bool = True
    ) -> dict:
        """
        Retrieve text chunks using hybrid BM25 + semantic search.

        Args:
            query: Text query string
            n_results: Number of results to return
            where: Optional metadata filter
            use_hybrid: Use hybrid retrieval (BM25 + semantic) vs semantic only

        Returns:
            Dictionary with ranked text results
        """
        if use_hybrid:
            # Get more results for fusion
            semantic_results = self._semantic_text_search(
                query, n_results=n_results * 3, where=where
            )
            bm25_scores = self._bm25_search(query)

            # Fuse scores
            fused_results = self._fuse_text_scores(
                semantic_results, bm25_scores, n_results
            )
            return fused_results
        else:
            # Semantic only
            return self._semantic_text_search(query, n_results, where)

    def retrieve_images(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[dict] = None
    ) -> dict:
        """
        Retrieve images using CLIP text-to-image search.

        Args:
            query: Text query string
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            Dictionary with ranked image results
        """
        # Embed query with CLIP text encoder
        query_embedding = self._clip_text_embed(query)

        # Query ChromaDB image collection
        results = self.db.query_images(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where
        )

        # Format results
        return self._format_image_results(results)

    def retrieve_multimodal(
        self,
        query: str,
        n_text: int = 5,
        n_images: int = 3,
        where: Optional[dict] = None
    ) -> dict:
        """
        Retrieve both text and images with unified ranking.

        Args:
            query: Text query string
            n_text: Number of text results
            n_images: Number of image results
            where: Optional metadata filter

        Returns:
            Dictionary with both text and image results
        """
        text_results = self.retrieve_text(query, n_text, where)
        image_results = self.retrieve_images(query, n_images, where)

        return {
            "text_results": text_results["results"],
            "image_results": image_results["results"],
            "query": query
        }

    def _semantic_text_search(
        self,
        query: str,
        n_results: int,
        where: Optional[dict] = None
    ) -> dict:
        """Semantic search using sentence transformers."""
        # Embed query
        query_embedding = self.text_model.encode([query], normalize_embeddings=True)[0]

        # Query ChromaDB
        results = self.db.query_text(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where
        )

        # Format results
        return self._format_text_results(results, score_type="semantic")

    def _bm25_search(self, query: str) -> np.ndarray:
        """BM25 keyword search."""
        query_tokens = self.bm25_indexer.tokenize_text(query)
        scores = self.bm25.get_scores(query_tokens)
        return scores

    def _fuse_text_scores(
        self,
        semantic_results: dict,
        bm25_scores: np.ndarray,
        n_results: int
    ) -> dict:
        """
        Fuse BM25 and semantic scores using weighted combination.

        Args:
            semantic_results: Results from semantic search
            bm25_scores: BM25 scores for all documents
            n_results: Number of final results

        Returns:
            Fused and re-ranked results
        """
        # Get semantic scores (convert distances to similarities)
        # ChromaDB returns squared L2 distances, convert to cosine similarity
        semantic_distances = np.array(semantic_results['distances'])
        semantic_scores = 1 / (1 + semantic_distances)  # Convert distance to similarity

        # Normalize BM25 scores to [0, 1]
        bm25_min = bm25_scores.min()
        bm25_max = bm25_scores.max()
        if bm25_max > bm25_min:
            bm25_normalized = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
        else:
            bm25_normalized = np.zeros_like(bm25_scores)

        # Get indices from semantic results
        result_indices = semantic_results['indices']

        # Compute fused scores for retrieved documents
        fused_scores = []
        for i, idx in enumerate(result_indices):
            semantic_score = semantic_scores[i]
            bm25_score = bm25_normalized[idx]
            fused_score = self.alpha * bm25_score + (1 - self.alpha) * semantic_score
            fused_scores.append((idx, fused_score))

        # Sort by fused score
        fused_scores.sort(key=lambda x: x[1], reverse=True)

        # Take top N results
        top_results = fused_scores[:n_results]

        # Format results
        results = {
            "results": [],
            "scores": [],
            "score_type": "hybrid"
        }

        for idx, score in top_results:
            # Find this result in semantic_results
            result_idx = result_indices.index(idx)
            results["results"].append(semantic_results['results'][result_idx])
            results["scores"].append(float(score))

        return results

    def _clip_text_embed(self, text: str) -> np.ndarray:
        """Embed text using CLIP text encoder."""
        import torch
        with torch.no_grad():
            inputs = self.clip_processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            text_features = self.clip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy()[0]

    def _format_text_results(self, results: dict, score_type: str = "semantic") -> dict:
        """Format text results from ChromaDB."""
        formatted = {
            "results": [],
            "scores": [],
            "score_type": score_type
        }

        # Store indices for later use
        formatted['indices'] = []
        formatted['distances'] = []

        for i in range(len(results['ids'][0])):
            formatted["results"].append({
                "chunk_id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            })
            # Convert distance to similarity (1 / (1 + distance))
            distance = results['distances'][0][i]
            similarity = 1 / (1 + distance)
            formatted["scores"].append(float(similarity))
            formatted['indices'].append(i)  # Store index
            formatted['distances'].append(distance)

        return formatted

    def _format_image_results(self, results: dict) -> dict:
        """Format image results from ChromaDB."""
        formatted = {
            "results": [],
            "scores": [],
            "score_type": "clip"
        }

        for i in range(len(results['ids'][0])):
            formatted["results"].append({
                "image_id": results['ids'][0][i],
                "image_path": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            })
            # Convert distance to similarity
            distance = results['distances'][0][i]
            similarity = 1 / (1 + distance)
            formatted["scores"].append(float(similarity))

        return formatted


def main():
    """Test multimodal retrieval."""
    # Initialize database
    db = MultimodalDB(persist_directory="../data/chroma_db")
    db.initialize_text_collection(
        chunks_path="../data/processed/chunks.json",
        embeddings_path="../data/embeddings/text_embeddings.pkl"
    )
    db.initialize_image_collection(
        metadata_path="../data/embeddings/image_metadata.json",
        embeddings_path="../data/embeddings/image_embeddings.pkl"
    )

    # Initialize retriever
    print("\n=== Initializing Retriever ===")
    retriever = MultimodalRetriever(
        db=db,
        alpha=0.4,  # 40% BM25, 60% semantic
        beta=0.7    # 70% text, 30% images
    )

    # Test text retrieval
    print("\n=== Test: Hybrid Text Retrieval ===")
    query = "artificial intelligence trends and developments"
    text_results = retriever.retrieve_text(query, n_results=3, use_hybrid=True)

    print(f"Query: '{query}'")
    print(f"Score type: {text_results['score_type']}")
    for i, (result, score) in enumerate(zip(text_results['results'], text_results['scores']), 1):
        print(f"\n{i}. Score: {score:.3f}")
        print(f"   Chunk: {result['chunk_id']}")
        print(f"   Article: {result['metadata']['article_title'][:50]}...")
        print(f"   Text: {result['text'][:100]}...")

    # Test image retrieval
    print("\n=== Test: Image Retrieval ===")
    img_results = retriever.retrieve_images(query, n_results=3)

    print(f"Query: '{query}'")
    for i, (result, score) in enumerate(zip(img_results['results'], img_results['scores']), 1):
        print(f"\n{i}. Score: {score:.3f}")
        print(f"   Image: {result['image_path']}")
        print(f"   Article: {result['metadata']['article_title'][:50]}...")

    # Test multimodal retrieval
    print("\n=== Test: Multimodal Retrieval ===")
    mm_results = retriever.retrieve_multimodal(
        query=query,
        n_text=3,
        n_images=2
    )

    print(f"Query: '{query}'")
    print(f"\nText results: {len(mm_results['text_results'])}")
    print(f"Image results: {len(mm_results['image_results'])}")

    # Test metadata filtering
    print("\n=== Test: Metadata Filtering ===")
    filtered_results = retriever.retrieve_text(
        query="technology",
        n_results=3,
        where={"article_id": 1}
    )

    print(f"Query: 'technology' (filtered to article_id=1)")
    for i, result in enumerate(filtered_results['results'], 1):
        print(f"{i}. {result['chunk_id']} - Article {result['metadata']['article_id']}")


if __name__ == "__main__":
    main()
