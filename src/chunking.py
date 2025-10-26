"""
Text chunking module for processing articles into semantic chunks.

Splits articles into optimal-sized chunks for embedding while preserving
context and associating relevant images with each chunk.
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter


class ArticleChunker:
    """Chunks articles into semantically coherent pieces with metadata."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[list[str]] = None
    ):
        """
        Initialize the article chunker.

        Args:
            chunk_size: Target size of each chunk in characters (~400-600 tokens)
            chunk_overlap: Overlap between chunks to maintain context
            separators: List of separators in priority order
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Default separators prioritize paragraph boundaries
        if separators is None:
            separators = [
                "\n\n",  # Paragraph breaks (highest priority)
                "\n",    # Line breaks
                ". ",    # Sentence boundaries
                " ",     # Word boundaries
                ""       # Character-level fallback
            ]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )

    def chunk_article(self, article: dict) -> list[dict]:
        """
        Chunk a single article into smaller pieces with metadata.

        Args:
            article: Article dictionary from scraper

        Returns:
            List of chunk dictionaries with metadata
        """
        article_id = article['article_id']
        article_title = article['title']
        article_url = article['url']
        article_date = article.get('date', None)
        article_categories = article.get('categories', [])
        content = article['content']
        images = article.get('images', [])

        # Convert ISO date to Unix timestamp for Qdrant range filtering
        article_timestamp = None
        if article_date:
            try:
                dt = datetime.fromisoformat(article_date.replace('Z', '+00:00'))
                article_timestamp = int(dt.timestamp())
            except Exception:
                pass

        # Split text into chunks
        text_chunks = self.text_splitter.split_text(content)

        # Create chunk objects with metadata
        chunks = []
        for idx, chunk_text in enumerate(text_chunks):
            # Calculate word count
            word_count = len(chunk_text.split())

            # Associate images with chunks
            associated_images = self._assign_images_to_chunk(
                chunk_index=idx,
                total_chunks=len(text_chunks),
                images=images
            )

            # Prepend article title to chunk text for better search
            searchable_text = f"{article_title}\n\n{chunk_text}"

            chunk = {
                'chunk_id': f"article_{article_id}_chunk_{idx}",
                'article_id': article_id,
                'article_title': article_title,
                'article_url': article_url,
                'article_date': article_date,
                'article_timestamp': article_timestamp,
                'article_categories': article_categories,
                'chunk_index': idx,
                'total_chunks': len(text_chunks),
                'chunk_text': searchable_text,
                'word_count': word_count,
                'associated_images': associated_images
            }

            chunks.append(chunk)

        return chunks

    def _assign_images_to_chunk(
        self,
        chunk_index: int,
        total_chunks: int,
        images: list[dict]
    ) -> list[str]:
        """
        Assign images to a specific chunk based on position.

        Strategy:
        - First chunk gets the first image (usually featured image)
        - Remaining images distributed evenly across chunks
        - Each chunk gets at most 2 images to avoid overload

        Args:
            chunk_index: Index of current chunk
            total_chunks: Total number of chunks in article
            images: List of image dictionaries from article

        Returns:
            List of image local_paths associated with this chunk
        """
        if not images:
            return []

        # Filter to only successfully downloaded images
        downloaded_images = [
            img['local_path'] for img in images
            if img.get('downloaded', False)
        ]

        if not downloaded_images:
            return []

        associated = []

        # First chunk always gets the first image (featured image)
        if chunk_index == 0:
            associated.append(downloaded_images[0])
            if len(downloaded_images) > 1:
                # Add one more image to first chunk if available
                associated.append(downloaded_images[1])

        # Distribute remaining images across other chunks
        elif len(downloaded_images) > 2:
            remaining_images = downloaded_images[2:]
            remaining_chunks = total_chunks - 1

            if remaining_chunks > 0 and remaining_images:
                # Calculate which images belong to this chunk
                images_per_chunk = len(remaining_images) / remaining_chunks
                start_idx = int((chunk_index - 1) * images_per_chunk)
                end_idx = int(chunk_index * images_per_chunk)

                chunk_images = remaining_images[start_idx:end_idx]
                associated.extend(chunk_images[:2])  # Max 2 images per chunk

        return associated

    def process_articles(
        self,
        input_path: str,
        output_path: str
    ) -> list[dict]:
        """
        Process all articles from input JSON and save chunks.

        Args:
            input_path: Path to scraped articles JSON
            output_path: Path to save processed chunks JSON

        Returns:
            List of all chunks
        """
        # Load articles
        with open(input_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)

        print(f"Processing {len(articles)} articles...")

        # Chunk all articles
        all_chunks = []
        for article in articles:
            chunks = self.chunk_article(article)
            all_chunks.extend(chunks)
            print(f"  Article {article['article_id']}: {article['title'][:50]}... "
                  f"-> {len(chunks)} chunks")

        # Save chunks
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)

        print(f"\nTotal chunks created: {len(all_chunks)}")
        print(f"Saved to: {output_path}")

        # Print statistics
        total_words = sum(chunk['word_count'] for chunk in all_chunks)
        avg_words = total_words / len(all_chunks) if all_chunks else 0
        chunks_with_images = sum(1 for c in all_chunks if c['associated_images'])

        print(f"\nStatistics:")
        print(f"  Average chunk size: {avg_words:.0f} words")
        print(f"  Chunks with images: {chunks_with_images}/{len(all_chunks)}")

        return all_chunks

    def apply_clip_matching(
        self,
        chunks_path: str,
        images_dir: str,
        threshold: float = 0.5
    ) -> list[dict]:
        """
        Apply CLIP-based image matching to existing chunks.

        Args:
            chunks_path: Path to chunks JSON
            images_dir: Images directory
            threshold: CLIP similarity threshold

        Returns:
            Updated chunks with CLIP assignments
        """
        from clip_image_matcher import CLIPImageMatcher

        # Load chunks
        with open(chunks_path, 'r') as f:
            chunks = json.load(f)

        print(f"\nApplying CLIP matching to {len(chunks)} chunks...")
        matcher = CLIPImageMatcher(threshold=threshold)
        images_path = Path(images_dir)

        # Group chunks by article
        from collections import defaultdict
        article_chunks = defaultdict(list)
        for chunk in chunks:
            article_chunks[chunk['article_id']].append(chunk)

        # Process each article's chunks
        updated_chunks = []
        for article_id, art_chunks in article_chunks.items():
            # Clear existing assignments
            for chunk in art_chunks:
                chunk['associated_images'] = []

            # Apply CLIP matching
            matched_chunks = matcher.match_images_to_chunks(
                chunks=art_chunks,
                images_dir=images_path,
                article_id=article_id
            )
            updated_chunks.extend(matched_chunks)

        # Save updated chunks
        with open(chunks_path, 'w') as f:
            json.dump(updated_chunks, f, indent=2, ensure_ascii=False)

        print(f"CLIP matching complete. Updated: {chunks_path}")
        return updated_chunks


def main():
    """CLI for chunking. Run: python chunking.py --help for options."""
    parser = argparse.ArgumentParser(description='Chunk articles and apply CLIP matching')
    parser.add_argument('--apply-clip', action='store_true', help='Apply CLIP matching to existing chunks')
    parser.add_argument('--threshold', type=float, default=0.5, help='CLIP similarity threshold')
    args = parser.parse_args()

    chunker = ArticleChunker(chunk_size=500, chunk_overlap=50)

    if args.apply_clip:
        chunks = chunker.apply_clip_matching(
            chunks_path='../data/processed/chunks.json',
            images_dir='../data/images',
            threshold=args.threshold
        )
    else:
        chunks = chunker.process_articles(
            input_path='../data/raw/articles_test_batch.json',
            output_path='../data/processed/chunks.json'
        )

    # Display sample chunks
    if chunks:
        print("\n=== Sample Chunks ===")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1}:")
            print(f"  ID: {chunk['chunk_id']}")
            print(f"  Article: {chunk['article_title'][:40]}...")
            print(f"  Words: {chunk['word_count']}")
            print(f"  Images: {len(chunk['associated_images'])}")
            print(f"  Text preview: {chunk['chunk_text'][:150]}...")


if __name__ == "__main__":
    main()
