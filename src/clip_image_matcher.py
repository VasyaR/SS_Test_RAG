"""
CLIP-based semantic image-to-chunk matching.

Uses CLIP to match images with text chunks based on semantic similarity.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPImageMatcher:
    """Matches images to text chunks using CLIP semantic similarity."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", threshold: float = 0.5):
        """
        Initialize CLIP matcher.

        Args:
            model_name: CLIP model from HuggingFace
            threshold: Similarity threshold after normalization (0-1)
        """
        self.threshold = threshold
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def match_images_to_chunks(self, chunks: list[dict], images_dir: Path, article_id: int) -> list[dict]:
        """
        Assign images to chunks using CLIP similarity.

        Args:
            chunks: Chunk dictionaries
            images_dir: Images directory
            article_id: Article ID

        Returns:
            Updated chunks with CLIP assignments
        """
        image_paths = list(images_dir.glob(f"article_{article_id}_img_*"))
        if not image_paths:
            return chunks

        chunk_texts = [c['chunk_text'] for c in chunks]
        similarities = self._compute_similarities(image_paths, chunk_texts)

        # Normalize to [0, 1]
        sim_min, sim_max = similarities.min(), similarities.max()
        if sim_max > sim_min:
            similarities = (similarities - sim_min) / (sim_max - sim_min)

        # Assign images to chunks above threshold
        for img_idx, img_path in enumerate(image_paths):
            for chunk_idx in range(len(chunks)):
                if similarities[img_idx, chunk_idx] >= self.threshold:
                    chunks[chunk_idx]['associated_images'].append(img_path.name)

        return chunks

    def _compute_similarities(self, image_paths: list[Path], texts: list[str]) -> np.ndarray:
        """Compute CLIP similarity matrix [n_images x n_chunks]."""
        with torch.no_grad():
            # Image features
            img_features = []
            for img_path in image_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    inputs = self.processor(images=img, return_tensors="pt")
                    feats = self.model.get_image_features(**inputs)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    img_features.append(feats.cpu().numpy()[0])
                except Exception:
                    img_features.append(np.zeros(512))

            # Text features
            txt_features = []
            for text in texts:
                inputs = self.processor(text=[text[:77]], return_tensors="pt", padding=True, truncation=True)
                feats = self.model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                txt_features.append(feats.cpu().numpy()[0])

            # Cosine similarity
            return np.dot(np.array(img_features), np.array(txt_features).T)
