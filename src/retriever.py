"""
Multimodal retrieval system combining BM25, semantic search, and CLIP.

Implements hybrid text retrieval and multimodal fusion for comprehensive
document and image retrieval.
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from qdrant_client.models import FieldCondition, Filter, MatchAny, Range
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from .database import MultimodalDB
from .embeddings import BM25Indexer


def build_filter(
    categories: list[str] | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    article_ids: list[int] | None = None
) -> Filter | None:
    """
    Build Qdrant filter from user-friendly parameters.

    Args:
        categories: List of category names to filter by
        date_start: ISO format date string (e.g., '2025-10-01')
        date_end: ISO format date string (e.g., '2025-11-01')
        article_ids: List of article IDs to filter by

    Returns:
        Qdrant Filter object or None if no filters specified
    """
    conditions = []

    if categories:
        conditions.append(
            FieldCondition(key="article_categories", match=MatchAny(any=categories))
        )

    if date_start or date_end:
        ts_start = int(datetime.fromisoformat(date_start).timestamp()) if date_start else None
        ts_end = int(datetime.fromisoformat(date_end).timestamp()) if date_end else None
        conditions.append(
            FieldCondition(key="article_timestamp", range=Range(gte=ts_start, lt=ts_end))
        )

    if article_ids:
        conditions.append(
            FieldCondition(key="article_id", match=MatchAny(any=article_ids))
        )

    return Filter(must=conditions) if conditions else None


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
        where: Filter | None = None,
        use_hybrid: bool = True
    ) -> dict:
        """
        Retrieve text chunks using hybrid BM25 + semantic search.

        Args:
            query: Text query string
            n_results: Number of results to return
            where: Optional Qdrant Filter object for metadata filtering
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
        where: Filter | None = None
    ) -> dict:
        """
        Retrieve images using CLIP text-to-image search.

        Args:
            query: Text query string
            n_results: Number of results to return
            where: Optional Qdrant Filter object for metadata filtering

        Returns:
            Dictionary with ranked image results
        """
        # Embed query with CLIP text encoder
        query_embedding = self._clip_text_embed(query)

        # Query Qdrant image collection
        results = self.db.query_images(
            query_embedding=query_embedding,
            n_results=n_results,
            query_filter=where
        )

        # Format results
        return self._format_image_results(results)

    def retrieve_multimodal(
        self,
        query: str,
        n_text: int = 5,
        n_images: int = 3,
        where: Filter | None = None
    ) -> dict:
        """
        Retrieve both text and images with smart ranking.

        Image strategy:
        1. First image: Best CLIP from best article (if article has images)
        2. Remaining images: General CLIP search (avoid duplicates)

        Args:
            query: Text query string
            n_text: Number of text results
            n_images: Number of image results
            where: Optional Qdrant Filter object for metadata filtering

        Returns:
            Dictionary with both text and image results
        """
        # Get text results
        text_results = self.retrieve_text(query, n_text, where)

        final_images = []
        used_image_ids = set()

        if text_results["results"]:
            # Try to get 1 image from best article
            best_article_id = text_results["results"][0]["metadata"]["article_id"]

            # Build filter for best article
            article_filter = build_filter(article_ids=[best_article_id])
            if where:
                # Combine with user's existing filters
                article_filter = Filter(must=where.must + article_filter.must)

            best_article_images = self.retrieve_images(
                query, n_results=1, where=article_filter
            )

            if best_article_images["results"]:
                first_img = best_article_images["results"][0]
                final_images.append(first_img)
                used_image_ids.add(first_img["image_id"])

        # Fill remaining slots with general CLIP search
        remaining = n_images - len(final_images)
        if remaining > 0:
            # Get more results to ensure we have enough after filtering duplicates
            general_results = self.retrieve_images(
                query, n_results=remaining + 5, where=where
            )

            for img in general_results["results"]:
                if img["image_id"] not in used_image_ids:
                    final_images.append(img)
                    used_image_ids.add(img["image_id"])
                    if len(final_images) >= n_images:
                        break

        return {
            "text_results": text_results["results"],
            "image_results": final_images,
            "query": query
        }

    def _semantic_text_search(
        self,
        query: str,
        n_results: int,
        where: Filter | None = None
    ) -> dict:
        """Semantic search using sentence transformers."""
        # Embed query
        query_embedding = self.text_model.encode([query], normalize_embeddings=True)[0]

        # Query Qdrant
        results = self.db.query_text(
            query_embedding=query_embedding,
            n_results=n_results,
            query_filter=where
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
        # Get semantic scores
        # Qdrant returns cosine similarities, convert to simple distance metric
        semantic_distances = np.array(semantic_results['distances'])
        semantic_scores = 1 - semantic_distances  # Convert distance back to similarity

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

        # Create mapping from index to position
        idx_to_pos = {idx: pos for pos, idx in enumerate(result_indices)}

        for idx, score in top_results:
            # Find this result in semantic_results
            result_pos = idx_to_pos.get(idx)
            if result_pos is not None:
                results["results"].append(semantic_results['results'][result_pos])
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

    def _format_text_results(self, results: list, score_type: str = "semantic") -> dict:
        """Format text results from Qdrant."""
        formatted = {
            "results": [],
            "scores": [],
            "score_type": score_type
        }

        # Store indices and distances for later use (for fusion)
        formatted['indices'] = []
        formatted['distances'] = []

        for i, scored_point in enumerate(results):
            formatted["results"].append({
                "chunk_id": scored_point.payload.get('chunk_id'),
                "text": scored_point.payload.get('chunk_text'),
                "metadata": {
                    "article_id": scored_point.payload.get('article_id'),
                    "article_title": scored_point.payload.get('article_title'),
                    "article_url": scored_point.payload.get('article_url'),
                    "article_date": scored_point.payload.get('article_date'),
                    "article_categories": scored_point.payload.get('article_categories', [])
                }
            })
            # Qdrant returns cosine similarity as score (0-1, higher is better)
            score = scored_point.score
            formatted["scores"].append(float(score))
            formatted['indices'].append(scored_point.id)
            # Convert similarity to distance for fusion compatibility
            distance = 1 - score
            formatted['distances'].append(distance)

        return formatted

    def _format_image_results(self, results: list) -> dict:
        """Format image results from Qdrant."""
        formatted = {
            "results": [],
            "scores": [],
            "score_type": "clip"
        }

        for scored_point in results:
            formatted["results"].append({
                "image_id": scored_point.id,
                "image_path": scored_point.payload.get('image_path'),
                "metadata": {
                    "article_id": scored_point.payload.get('article_id'),
                    "article_title": scored_point.payload.get('article_title'),
                    "full_path": scored_point.payload.get('full_path')
                }
            })
            # Qdrant returns cosine similarity as score
            formatted["scores"].append(float(scored_point.score))

        return formatted


def main():
    """Test multimodal retrieval."""
    # Initialize database
    db = MultimodalDB(persist_directory="../data/qdrant_db")

    # Initialize retriever
    print("\n=== Initializing Retriever ===")
    retriever = MultimodalRetriever(
        db=db,
        alpha=0.4,  # 40% BM25, 60% semantic
        beta=0.7    # 70% text, 30% images
    )

    # Test text retrieval
    print("\n=== Test 1: Hybrid Text Retrieval ===")
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
    print("\n=== Test 2: Image Retrieval ===")
    img_results = retriever.retrieve_images(query, n_results=3)

    print(f"Query: '{query}'")
    for i, (result, score) in enumerate(zip(img_results['results'], img_results['scores']), 1):
        print(f"\n{i}. Score: {score:.3f}")
        print(f"   Image: {result['image_path']}")
        print(f"   Article: {result['metadata']['article_title'][:50]}...")

    # Test multimodal retrieval
    print("\n=== Test 3: Multimodal Retrieval ===")
    mm_results = retriever.retrieve_multimodal(
        query=query,
        n_text=3,
        n_images=2
    )

    print(f"Query: '{query}'")
    print(f"\nText results: {len(mm_results['text_results'])}")
    print(f"Image results: {len(mm_results['image_results'])}")

    # Test category filtering
    print("\n=== Test 4: Category Filtering ===")
    category_filter = build_filter(categories=["ML Research", "Business"])
    filtered_results = retriever.retrieve_text(
        query="machine learning",
        n_results=3,
        where=category_filter
    )

    print(f"Query: 'machine learning' (filtered to ML Research/Business)")
    for i, result in enumerate(filtered_results['results'], 1):
        print(f"{i}. {result['chunk_id']}")
        print(f"   Categories: {result['metadata']['article_categories']}")

    # Test date filtering
    print("\n=== Test 5: Date Range Filtering (October 2025) ===")
    date_filter = build_filter(date_start="2025-10-01", date_end="2025-11-01")
    date_results = retriever.retrieve_text(
        query="AI news",
        n_results=3,
        where=date_filter
    )

    print(f"Query: 'AI news' (October 2025 only)")
    for i, result in enumerate(date_results['results'], 1):
        print(f"{i}. {result['metadata']['article_title'][:40]}...")
        print(f"   Date: {result['metadata']['article_date'][:10]}")


if __name__ == "__main__":
    main()
