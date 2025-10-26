"""
Image embedding generation using CLIP for visual-semantic retrieval.

Handles CLIP-based image embeddings with support for both image and text queries
in the same embedding space.
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPImageEmbedder:
    """Generate image embeddings using CLIP."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None
    ):
        """
        Initialize CLIP image embedder.

        Args:
            model_name: HuggingFace CLIP model name
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading CLIP model: {model_name}...")
        print(f"Using device: {self.device}")

        # Load CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Get embedding dimension
        self.embedding_dim = self.model.config.projection_dim
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def embed_images(
        self,
        image_paths: list[str],
        batch_size: int = 8
    ) -> tuple[np.ndarray, list[str]]:
        """
        Generate embeddings for a list of images.

        Args:
            image_paths: List of image file paths
            batch_size: Batch size for processing

        Returns:
            Tuple of (embeddings array, list of successfully processed paths)
        """
        embeddings = []
        valid_paths = []

        print(f"Processing {len(image_paths)} images...")

        with torch.no_grad():
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch_images = []
                batch_valid_paths = []

                # Load images in batch
                for img_path in batch_paths:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        batch_images.append(img)
                        batch_valid_paths.append(img_path)
                    except Exception as e:
                        print(f"  Warning: Failed to load {img_path}: {e}")
                        continue

                if not batch_images:
                    continue

                # Process batch
                inputs = self.processor(
                    images=batch_images,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                # Get image embeddings
                image_features = self.model.get_image_features(**inputs)

                # Normalize embeddings (for cosine similarity)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Move to CPU and convert to numpy
                batch_embeddings = image_features.cpu().numpy()
                embeddings.append(batch_embeddings)
                valid_paths.extend(batch_valid_paths)

                print(f"  Processed {len(valid_paths)}/{len(image_paths)} images")

        if embeddings:
            embeddings = np.vstack(embeddings)
        else:
            embeddings = np.array([])

        return embeddings, valid_paths

    def embed_text(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for text queries using CLIP text encoder.

        Args:
            texts: List of text strings

        Returns:
            Numpy array of text embeddings
        """
        with torch.no_grad():
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            # Get text embeddings
            text_features = self.model.get_text_features(**inputs)

            # Normalize embeddings
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            return text_features.cpu().numpy()

    def process_chunks(
        self,
        chunks_path: str,
        images_dir: str,
        embeddings_output: str,
        metadata_output: str,
        articles_path: str = "../data/raw/articles_test_batch.json"
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Process all images from images directory and generate embeddings.

        Args:
            chunks_path: Path to chunks JSON
            images_dir: Directory containing images
            embeddings_output: Path to save embeddings
            metadata_output: Path to save metadata
            articles_path: Path to original articles JSON with full metadata

        Returns:
            Tuple of (embeddings array, metadata list)
        """
        # Load original articles for full metadata (categories, dates)
        with open(articles_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)

        # Build article mapping from original articles
        from datetime import datetime
        article_map = {}
        for article in articles:
            article_id = article['article_id']
            # Parse date to timestamp
            date_str = article.get('date', '')
            timestamp = None
            if date_str:
                try:
                    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    timestamp = int(dt.timestamp())
                except:
                    pass

            article_map[article_id] = {
                'title': article.get('title', ''),
                'url': article.get('url', ''),
                'date': date_str,
                'timestamp': timestamp,
                'categories': article.get('categories', [])
            }

        # Process ALL images in directory
        images_path = Path(images_dir)
        all_images = sorted(images_path.glob('article_*_img_*'))

        image_metadata = []
        for img_path in all_images:
            # Extract article_id from filename (e.g., article_5_img_2.jpg -> 5)
            filename = img_path.name
            try:
                article_id = int(filename.split('_')[1])
                article_info = article_map.get(article_id, {})

                image_metadata.append({
                    'image_path': filename,
                    'full_path': str(img_path),
                    'article_id': article_id,
                    'article_title': article_info.get('title', 'Unknown'),
                    'article_url': article_info.get('url', ''),
                    'article_date': article_info.get('date', ''),
                    'article_timestamp': article_info.get('timestamp'),
                    'article_categories': article_info.get('categories', [])
                })
            except (IndexError, ValueError):
                continue

        print(f"Found {len(image_metadata)} images in directory")

        # Generate embeddings
        image_paths = [meta['full_path'] for meta in image_metadata]
        embeddings, valid_paths = self.embed_images(image_paths)

        # Filter metadata to only valid images
        valid_metadata = [
            meta for meta in image_metadata
            if meta['full_path'] in valid_paths
        ]

        # Save embeddings
        embeddings_file = Path(embeddings_output)
        embeddings_file.parent.mkdir(parents=True, exist_ok=True)

        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings, f)

        print(f"Image embeddings saved to: {embeddings_output}")
        print(f"Shape: {embeddings.shape}")

        # Save metadata
        metadata_file = Path(metadata_output)
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(valid_metadata, f, indent=2, ensure_ascii=False)

        print(f"Image metadata saved to: {metadata_output}")

        return embeddings, valid_metadata


def main():
    """Test CLIP image embeddings on article images."""
    # Initialize CLIP embedder
    embedder = CLIPImageEmbedder(model_name="openai/clip-vit-base-patch32")

    # Process images
    print("\n=== Processing Images ===")
    embeddings, metadata = embedder.process_chunks(
        chunks_path="../data/processed/chunks.json",
        images_dir="../data/images",
        embeddings_output="../data/embeddings/image_embeddings.pkl",
        metadata_output="../data/embeddings/image_metadata.json"
    )

    # Test text-to-image retrieval
    print("\n=== Test: Text-to-Image Retrieval ===")
    text_queries = [
        "artificial intelligence",
        "computer technology",
        "data visualization"
    ]

    for query in text_queries:
        print(f"\nQuery: '{query}'")

        # Embed query text
        query_embedding = embedder.embed_text([query])

        # Compute similarities
        similarities = np.dot(embeddings, query_embedding.T).flatten()

        # Get top 3 results
        top_3_idx = np.argsort(similarities)[::-1][:3]

        print("Top 3 results:")
        for rank, idx in enumerate(top_3_idx, 1):
            meta = metadata[idx]
            score = similarities[idx]
            print(f"  {rank}. Score: {score:.3f}")
            print(f"     Image: {meta['image_path']}")
            print(f"     Article: {meta['article_title'][:50]}...")

    # Test image-to-image retrieval
    print("\n=== Test: Image-to-Image Retrieval ===")
    if len(metadata) > 0:
        # Use first image as query
        query_img_path = metadata[0]['full_path']
        print(f"Query image: {metadata[0]['image_path']}")

        # Embed query image
        query_embedding, _ = embedder.embed_images([query_img_path])

        # Compute similarities (skip the query image itself)
        similarities = np.dot(embeddings, query_embedding.T).flatten()

        # Get top 4 results (excluding itself at rank 1)
        top_4_idx = np.argsort(similarities)[::-1][:4]

        print("\nTop 3 similar images:")
        for rank, idx in enumerate(top_4_idx[1:], 1):  # Skip first (itself)
            meta = metadata[idx]
            score = similarities[idx]
            print(f"  {rank}. Score: {score:.3f}")
            print(f"     Image: {meta['image_path']}")
            print(f"     Article: {meta['article_title'][:50]}...")


if __name__ == "__main__":
    main()
