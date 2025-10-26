"""
Web scraper for The Batch articles from DeepLearning.AI.

This module scrapes articles from https://www.deeplearning.ai/the-batch/
including text content, metadata, and associated images.
"""

import argparse
import json
import os
import time
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image
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

        # Category tab URLs
        self.category_urls = {
            'Weekly Issues': base_url,
            "Andrew's Letters": f"{base_url}tag/letters/",
            'Data Points': f"{base_url}tag/data-points/",
            'ML Research': f"{base_url}tag/research/",
            'Business': f"{base_url}tag/business/",
            'Science': f"{base_url}tag/science/",
            'Culture': f"{base_url}tag/culture/",
            'Hardware': f"{base_url}tag/hardware/",
            'AI Careers': f"{base_url}tag/ai-careers/"
        }

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

    def scrape_article_list(
        self,
        max_articles: int = 20,
        max_pages: int = 5,
        category_url: Optional[str] = None
    ) -> list[dict]:
        """
        Scrape list of articles from The Batch, checking multiple pages if needed.

        Args:
            max_articles: Maximum number of VALID articles to collect
            max_pages: Maximum number of pages to check
            category_url: Optional category URL to scrape from (e.g., /tag/research/)

        Returns:
            List of article metadata dictionaries
        """
        all_articles = []
        seen_urls = set()

        base = category_url if category_url else self.base_url

        for page in range(1, max_pages + 1):
            if len(all_articles) >= max_articles:
                break

            if page == 1:
                url = base
            else:
                url = f"{base}page/{page}/"

            print(f"Fetching article list from page {page}: {url}")

            try:
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
            except requests.RequestException as e:
                print(f"  Error fetching page: {e}")
                break

            soup = BeautifulSoup(response.content, 'lxml')

            # Find all links that look like articles
            all_links = soup.find_all('a', href=True)
            page_articles = 0

            for link in all_links:
                if len(all_articles) >= max_articles:
                    break

                href = link['href']
                if not href.startswith('http'):
                    href = urljoin(self.base_url, href)

                # Filter: must be /the-batch/ article, not tag/category/home/about
                if ('/the-batch/' in href and
                    href != self.base_url and
                    '/tag/' not in href and
                    '/category/' not in href and
                    '/about/' not in href and
                    href not in seen_urls):

                    title = link.get_text(strip=True) or "Untitled"
                    all_articles.append({
                        'url': href,
                        'title': title
                    })
                    seen_urls.add(href)
                    page_articles += 1

            print(f"  Found {page_articles} new articles on page {page} (total: {len(all_articles)})")

            # Stop if no articles found on this page
            if page_articles == 0:
                print(f"  No more articles found, stopping pagination")
                break

        print(f"\nCollected {len(all_articles)} unique article URLs")
        return all_articles

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

        # Extract date (The Batch uses JSON-LD)
        date = None

        # Try JSON-LD first (most reliable for The Batch)
        try:
            json_ld_scripts = soup.find_all('script', type='application/ld+json')
            for script in json_ld_scripts:
                if script.string:
                    data = json.loads(script.string)
                    if 'datePublished' in data:
                        date = data['datePublished']
                        break
        except Exception as e:
            pass

        # Try <time> tag
        if not date:
            date_elem = soup.find('time')
            if date_elem:
                date = date_elem.get('datetime') or date_elem.get_text(strip=True)

        # Try meta tags
        if not date:
            meta_date = soup.find('meta', {'property': 'article:published_time'}) or \
                       soup.find('meta', {'name': 'publish_date'}) or \
                       soup.find('meta', {'property': 'og:published_time'})
            if meta_date:
                date = meta_date.get('content')

        # Try class-based selectors
        if not date:
            date_elem = soup.find(class_=lambda x: x and any(d in x.lower() for d in ['date', 'publish', 'time']))
            if date_elem:
                date = date_elem.get_text(strip=True)

        # Extract categories (multiple possible per article)
        categories = set()

        # Category mapping based on tag URLs
        category_map = {
            '/tag/letters/': "Andrew's Letters",
            '/tag/data-points/': 'Data Points',
            '/tag/research/': 'ML Research',
            '/tag/business/': 'Business',
            '/tag/science/': 'Science',
            '/tag/culture/': 'Culture',
            '/tag/hardware/': 'Hardware',
            '/tag/ai-careers/': 'AI Careers'
        }

        # Default: Weekly Issues for issue-XXX pattern
        if '/issue-' in url:
            categories.add('Weekly Issues')

        # Check for category tags in the article
        article_tag = soup.find('article')
        if article_tag:
            tag_links = article_tag.find_all('a', href=lambda x: x and '/tag/' in x)
            for link in tag_links:
                href = link['href']
                for tag_pattern, category_name in category_map.items():
                    if tag_pattern in href:
                        categories.add(category_name)

        # Convert to sorted list for consistent ordering
        categories = sorted(list(categories)) if categories else ['Uncategorized']

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
            'categories': categories,
            'content': content_text,
            'images': images,
            'tags': tags,
            'word_count': len(content_text.split())
        }

        return article_data

    def download_image(self, image_url: str, save_path: Path) -> bool:
        """
        Download and save an image with proper format handling.

        Args:
            image_url: URL of the image
            save_path: Path to save the image

        Returns:
            True if successful, False otherwise
        """
        try:
            # Skip data: URIs (inline SVG/GIF placeholders)
            if image_url.startswith('data:'):
                return False

            response = self.session.get(image_url, timeout=10)
            response.raise_for_status()

            # Open and process image
            img = Image.open(BytesIO(response.content))

            # Convert RGBA/P to RGB for JPEG, or save as PNG
            if img.mode in ('RGBA', 'P', 'LA'):
                # Change extension to .png
                save_path = save_path.with_suffix('.png')
                # Convert P (palette) to RGBA first for better quality
                if img.mode == 'P':
                    img = img.convert('RGBA')
                img.save(save_path, 'PNG')
            else:
                # RGB images can be saved as JPEG
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(save_path, 'JPEG', quality=90)

            return True
        except Exception as e:
            # Only print non-trivial errors
            if 'data:image' not in str(e):
                print(f"    Failed to download image: {e}")
            return False

    def scrape_batch(
        self,
        num_articles: int = 20,
        batch_name: str = "batch_1",
        incremental: bool = False
    ) -> list[dict]:
        """
        Scrape a batch of articles with optional incremental mode.

        Args:
            num_articles: Number of articles to scrape
            batch_name: Name for the output JSON file
            incremental: If True, skip articles that already exist in database

        Returns:
            List of scraped article dictionaries
        """
        print(f"\n=== Starting scrape: {num_articles} articles (incremental={incremental}) ===\n")

        # Load existing articles if incremental
        existing_urls = set()
        existing_articles = []
        output_path = self.raw_dir / f"articles_{batch_name}.json"

        if incremental and output_path.exists():
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_articles = json.load(f)
                existing_urls = {art['url'] for art in existing_articles}
                print(f"Found {len(existing_articles)} existing articles in database")
            except Exception as e:
                print(f"Warning: Could not load existing articles: {e}")

        # Get article URLs
        article_list = self.scrape_article_list(max_articles=num_articles, max_pages=5)

        if not article_list:
            print("No articles found!")
            return existing_articles

        # Scrape each article
        scraped_articles = []
        next_article_id = len(existing_articles)  # Continue from last ID

        for article_meta in tqdm(article_list, desc="Scraping articles"):
            url = article_meta['url']

            # Skip if URL already exists
            if incremental and url in existing_urls:
                print(f"\n  Skipping (already exists): {url}")
                continue

            article_data = self.scrape_article_content(
                url=url,
                article_id=next_article_id
            )

            if article_data:
                # Download images for this article
                for img_info in article_data['images']:
                    img_path = self.images_dir / img_info['local_path']
                    success = self.download_image(img_info['url'], img_path)
                    img_info['downloaded'] = success

                    # Update local_path if extension changed (RGBA/P saved as PNG)
                    if success:
                        # Check if file was saved with different extension
                        actual_path = img_path.with_suffix('.png') if (self.images_dir / img_path.with_suffix('.png').name).exists() else img_path
                        img_info['local_path'] = actual_path.name

                scraped_articles.append(article_data)
                existing_urls.add(url)
                next_article_id += 1

        # Combine existing and new articles
        all_articles = existing_articles + scraped_articles

        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_articles, f, indent=2, ensure_ascii=False)

        print(f"\n=== Scraping complete ===")
        print(f"New articles scraped: {len(scraped_articles)}")
        print(f"Total articles in database: {len(all_articles)}")
        print(f"Saved to: {output_path}")

        return all_articles

    def scrape_multi_category(
        self,
        articles_per_category: int = 3,
        categories: Optional[list[str]] = None,
        batch_name: str = "batch_1",
        incremental: bool = False
    ) -> list[dict]:
        """
        Scrape articles from multiple categories with balanced distribution.

        Args:
            articles_per_category: Number of articles to scrape per category
            categories: List of category names to scrape (None = all categories)
            batch_name: Name for the output JSON file
            incremental: If True, skip articles that already exist in database

        Returns:
            List of scraped article dictionaries
        """
        print(f"\n=== Multi-Category Scraping ===")
        print(f"Target: {articles_per_category} articles per category")
        print(f"Incremental mode: {incremental}\n")

        # Load existing articles if incremental
        existing_urls = set()
        existing_articles = []
        output_path = self.raw_dir / f"articles_{batch_name}.json"

        if incremental and output_path.exists():
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_articles = json.load(f)
                existing_urls = {art['url'] for art in existing_articles}
                print(f"Found {len(existing_articles)} existing articles in database\n")
            except Exception as e:
                print(f"Warning: Could not load existing articles: {e}\n")

        # Determine which categories to scrape
        target_categories = categories if categories else list(self.category_urls.keys())

        scraped_articles = []
        next_article_id = len(existing_articles)

        for category_name in target_categories:
            if category_name not in self.category_urls:
                print(f"Warning: Unknown category '{category_name}', skipping")
                continue

            category_url = self.category_urls[category_name]
            print(f"=== Scraping {category_name} ===")

            # Get article URLs from this category
            article_list = self.scrape_article_list(
                max_articles=articles_per_category * 3,  # Get extra for filtering
                max_pages=3,
                category_url=category_url
            )

            # Scrape articles from this category
            category_count = 0
            for article_meta in article_list:
                if category_count >= articles_per_category:
                    break

                url = article_meta['url']

                # Skip if URL already exists
                if incremental and url in existing_urls:
                    print(f"  Skipping (already exists): {url}")
                    continue

                article_data = self.scrape_article_content(
                    url=url,
                    article_id=next_article_id
                )

                if article_data:
                    # Download images
                    for img_info in article_data['images']:
                        img_path = self.images_dir / img_info['local_path']
                        success = self.download_image(img_info['url'], img_path)
                        img_info['downloaded'] = success

                        # Update local_path if extension changed
                        if success:
                            actual_path = img_path.with_suffix('.png') if (self.images_dir / img_path.with_suffix('.png').name).exists() else img_path
                            img_info['local_path'] = actual_path.name

                    scraped_articles.append(article_data)
                    existing_urls.add(url)
                    next_article_id += 1
                    category_count += 1

            print(f"Scraped {category_count} articles from {category_name}\n")

        # Combine existing and new articles
        all_articles = existing_articles + scraped_articles

        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_articles, f, indent=2, ensure_ascii=False)

        # Print summary
        print(f"\n=== Multi-Category Scraping Complete ===")
        print(f"New articles scraped: {len(scraped_articles)}")
        print(f"Total articles in database: {len(all_articles)}")

        # Category distribution
        from collections import Counter
        category_counts = Counter()
        for article in all_articles:
            for category in article.get('categories', []):
                category_counts[category] += 1

        print(f"\nCategory distribution:")
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count} articles")

        print(f"\nSaved to: {output_path}")

        return all_articles


def main():
    """CLI for scraping. Run: python scraper.py --help for options."""
    parser = argparse.ArgumentParser(description='Scrape The Batch articles')
    parser.add_argument('--multi-category', action='store_true', help='Multi-category scraping')
    parser.add_argument('--per-category', type=int, default=2, help='Articles per category (multi mode)')
    parser.add_argument('--categories', nargs='+', help='Specific categories (optional)')
    parser.add_argument('--incremental', action='store_true', help='Skip existing articles')
    args = parser.parse_args()

    scraper = TheBatchScraper(output_dir="../data")

    if args.multi_category:
        articles = scraper.scrape_multi_category(
            articles_per_category=args.per_category,
            categories=args.categories,
            batch_name="test_batch",
            incremental=args.incremental
        )
    else:
        articles = scraper.scrape_batch(
            num_articles=20,
            batch_name="test_batch",
            incremental=args.incremental
        )

        # Show category distribution for standard mode
        category_counts = Counter()
        for article in articles:
            for category in article.get('categories', []):
                category_counts[category] += 1

        print(f"\nCategory distribution:")
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count} articles")


if __name__ == "__main__":
    main()
