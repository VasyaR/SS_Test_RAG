"""
Web scraper for The Batch articles from DeepLearning.AI.

This module scrapes articles from https://www.deeplearning.ai/the-batch/
including text content, metadata, and associated images.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from tqdm import tqdm


class TheBatchScraper:
    """Scraper for The Batch articles."""

    def __init__(
        self,
        base_url: str = "https://www.deeplearning.ai/the-batch/",
        output_dir: str = "../data",
        delay: float = 1.5
    ):
        """
        Initialize The Batch scraper.

        Args:
            base_url: Base URL for The Batch
            output_dir: Output directory for scraped data
            delay: Delay between requests in seconds (respectful scraping)
        """
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.delay = delay

        # Create output directories
        self.raw_dir = self.output_dir / "raw"
        self.images_dir = self.output_dir / "images"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational RAG Project)'
        })

    def scrape_article_list(self, page: int = 1, max_articles: int = 20) -> list[dict]:
        """
        Scrape list of articles from The Batch homepage or paginated page.

        Args:
            page: Page number to scrape
            max_articles: Maximum number of articles to collect

        Returns:
            List of article metadata dictionaries
        """
        articles = []

        if page == 1:
            url = self.base_url
        else:
            url = f"{self.base_url}page/{page}/"

        print(f"Fetching article list from: {url}")

        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching article list: {e}")
            return articles

        soup = BeautifulSoup(response.content, 'lxml')

        # Find article links - The Batch uses various article structures
        # Look for article titles with links
        article_elements = soup.find_all('article') or soup.find_all('div', class_=lambda x: x and 'post' in x.lower())

        if not article_elements:
            # Fallback: find all links that look like articles
            all_links = soup.find_all('a', href=True)
            article_links = [
                a for a in all_links
                if '/the-batch/' in a['href'] and a['href'] != self.base_url
            ]

            for link in article_links[:max_articles]:
                href = link['href']
                if not href.startswith('http'):
                    href = urljoin(self.base_url, href)

                title = link.get_text(strip=True) or "Untitled"

                articles.append({
                    'url': href,
                    'title': title
                })
        else:
            # Extract from article elements
            for article in article_elements[:max_articles]:
                link = article.find('a', href=True)
                if not link:
                    continue

                href = link['href']
                if not href.startswith('http'):
                    href = urljoin(self.base_url, href)

                # Get title
                title_elem = article.find(['h1', 'h2', 'h3'])
                title = title_elem.get_text(strip=True) if title_elem else link.get_text(strip=True)

                articles.append({
                    'url': href,
                    'title': title
                })

        # Remove duplicates
        seen_urls = set()
        unique_articles = []
        for article in articles:
            if article['url'] not in seen_urls:
                seen_urls.add(article['url'])
                unique_articles.append(article)

        print(f"Found {len(unique_articles)} unique article URLs")
        return unique_articles[:max_articles]

    def scrape_article_content(self, url: str, article_id: int) -> Optional[dict]:
        """
        Scrape full content of a single article.

        Args:
            url: Article URL
            article_id: Unique ID for the article

        Returns:
            Dictionary with article data or None if failed
        """
        time.sleep(self.delay)  # Respectful scraping

        print(f"  Scraping: {url}")

        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"  Error fetching article: {e}")
            return None

        soup = BeautifulSoup(response.content, 'lxml')

        # Extract title
        title = None
        title_elem = soup.find('h1') or soup.find('title')
        if title_elem:
            title = title_elem.get_text(strip=True)

        # Extract date
        date = None
        date_elem = soup.find('time') or soup.find(class_=lambda x: x and 'date' in x.lower())
        if date_elem:
            date = date_elem.get_text(strip=True) or date_elem.get('datetime')

        # Extract main content
        content_paragraphs = []

        # Look for main content area
        main_content = soup.find('article') or soup.find('main') or soup.find(class_=lambda x: x and 'content' in x.lower())

        if main_content:
            paragraphs = main_content.find_all('p')
        else:
            paragraphs = soup.find_all('p')

        for p in paragraphs:
            text = p.get_text(strip=True)
            if len(text) > 20:  # Filter out very short paragraphs
                content_paragraphs.append(text)

        content_text = '\n\n'.join(content_paragraphs)

        # Extract images
        images = []
        img_elements = soup.find_all('img', src=True)

        for idx, img in enumerate(img_elements):
            src = img['src']
            if not src.startswith('http'):
                src = urljoin(url, src)

            # Skip very small images (likely icons)
            if 'icon' in src.lower() or 'logo' in src.lower():
                continue

            alt_text = img.get('alt', '')

            images.append({
                'url': src,
                'alt_text': alt_text,
                'local_path': f"article_{article_id}_img_{idx}.jpg"
            })

        # Extract tags
        tags = []
        tag_elements = soup.find_all('a', class_=lambda x: x and 'tag' in x.lower())
        for tag_elem in tag_elements:
            tags.append(tag_elem.get_text(strip=True))

        if not title or not content_text:
            print(f"  Skipping article (missing critical fields)")
            return None

        article_data = {
            'article_id': article_id,
            'url': url,
            'title': title,
            'date': date,
            'content': content_text,
            'images': images,
            'tags': tags,
            'word_count': len(content_text.split())
        }

        return article_data

    def download_image(self, image_url: str, save_path: Path) -> bool:
        """
        Download and save an image.

        Args:
            image_url: URL of the image
            save_path: Path to save the image

        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.session.get(image_url, timeout=10)
            response.raise_for_status()

            # Verify it's an image
            img = Image.open(BytesIO(response.content))
            img.save(save_path)
            return True
        except Exception as e:
            print(f"    Failed to download image: {e}")
            return False

    def scrape_batch(
        self,
        num_articles: int = 20,
        start_page: int = 1,
        batch_name: str = "batch_1"
    ) -> list[dict]:
        """
        Scrape a batch of articles.

        Args:
            num_articles: Number of articles to scrape
            start_page: Starting page number
            batch_name: Name for the output JSON file

        Returns:
            List of scraped article dictionaries
        """
        print(f"\n=== Starting scrape: {num_articles} articles ===\n")

        # Get article URLs
        article_list = self.scrape_article_list(page=start_page, max_articles=num_articles)

        if not article_list:
            print("No articles found!")
            return []

        # Scrape each article
        scraped_articles = []

        for idx, article_meta in enumerate(tqdm(article_list, desc="Scraping articles")):
            article_data = self.scrape_article_content(
                url=article_meta['url'],
                article_id=idx
            )

            if article_data:
                # Download images for this article
                for img_info in article_data['images']:
                    img_path = self.images_dir / img_info['local_path']
                    success = self.download_image(img_info['url'], img_path)
                    img_info['downloaded'] = success

                scraped_articles.append(article_data)

        # Save to JSON
        output_path = self.raw_dir / f"articles_{batch_name}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scraped_articles, f, indent=2, ensure_ascii=False)

        print(f"\n=== Scraping complete ===")
        print(f"Articles scraped: {len(scraped_articles)}")
        print(f"Saved to: {output_path}")

        return scraped_articles


def main():
    """Main function for testing the scraper."""
    # Initialize scraper (use relative path from src/)
    scraper = TheBatchScraper(output_dir="../data")

    # Scrape 10-20 articles for testing
    articles = scraper.scrape_batch(num_articles=15, batch_name="test_batch")

    # Print summary
    print(f"\n=== Summary ===")
    for article in articles:
        print(f"- {article['title'][:60]}... ({article['word_count']} words, {len(article['images'])} images)")


if __name__ == "__main__":
    main()
