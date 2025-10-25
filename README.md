# Multimodal RAG System for The Batch

A Retrieval-Augmented Generation (RAG) system that retrieves relevant news articles from [The Batch](https://www.deeplearning.ai/the-batch/) using both textual and visual data. Users can query the system and receive AI-generated answers along with relevant articles and images.

## Project Overview

This system enables semantic search over The Batch articles by combining:
- **Text retrieval**: BM25 (keyword-based) + Sentence Transformers (semantic)
- **Image retrieval**: CLIP embeddings for visual content
- **Multimodal fusion**: Combines text and image similarity scores
- **LLM-powered answers**: Local LLM generates answers from retrieved context
- **Interactive UI**: Gradio interface for easy querying

## Technology Stack

### Core Components (100% Free/Local)

#### Web Scraping
- **BeautifulSoup4**: Parse HTML from The Batch
- **Requests**: Fetch web pages
- **Why**: Simple, reliable scraping for static content

#### Text Processing
- **sentence-transformers** (`all-MiniLM-L6-v2`): Generate text embeddings
  - **Why**: Fast, lightweight, good quality embeddings (384 dims)
- **rank_bm25**: Traditional keyword search
  - **Why**: Complements semantic search, catches exact matches
- **nltk**: Tokenization
  - **Why**: Standard library for text processing

#### Image Processing
- **CLIP** (`openai/clip-vit-base-patch32`): Generate image embeddings
  - **Why**: Open-source multimodal model, trained on image-text pairs

#### Vector Database
- **ChromaDB**: Store and query embeddings with metadata
  - **Why**: Local, supports metadata filtering, easy to use, no server needed

#### LLM
- **Ollama** (Llama 3.2 or Mistral): Local language model
  - **Why**: Free, runs locally, no API costs, good quality
- **Alternative**: HuggingFace Transformers
  - **Why**: Fallback if Ollama not available

#### Chunking
- **LangChain TextSplitter** or custom paragraph-based splitter
  - **Why**: Maintains semantic coherence, 200-500 tokens optimal for embeddings

#### UI
- **Gradio**: Interactive web interface
  - **Why**: Simple, fast to build, supports images/galleries, used in existing RAG/

### Architecture

```
User Query
    ↓
Query Embedding (Text + optional Image)
    ↓
Hybrid Retrieval (BM25 + Semantic + CLIP)
    ↓
Metadata Filtering (date, count, etc.)
    ↓
Top K Chunks + Images + Articles
    ↓
LLM Context Generation
    ↓
Answer + Retrieved Articles + Images
    ↓
Gradio UI Display
```

## Implementation Approach

### Chunking Strategy
- **Method**: Paragraph-based with 200-500 token chunks
- **Why**: Articles are well-structured with clear paragraphs
- **Overlap**: 50 tokens to maintain context across chunks
- **Image linking**: Associate images with nearest text chunks based on HTML structure

### Multimodal Fusion
- **Text score**: `α * BM25_score + (1-α) * Semantic_score`
- **Image score**: CLIP similarity between query and image embeddings
- **Final score**: `β * Text_score + (1-β) * Image_score`
- **Parameters**: User-adjustable in UI (like existing RAG/ system)

### Metadata Filtering
- Stored metadata: `date`, `title`, `article_id`, `url`, `image_path`, `tags`
- ChromaDB `where` clause for filtering before/after retrieval
- Examples: "2 latest articles", "articles from 2024", "articles about transformers"

## Project Structure

```
├── data/                           # Data storage (gitignored)
│   ├── raw/                        # Raw scraped articles (JSON)
│   ├── images/                     # Downloaded images
│   ├── processed/                  # Processed chunks
│   ├── embeddings/                 # Saved embeddings (split files)
│   └── cache/                      # BM25, tokenized docs cache
├── src/                            # Source code
│   ├── scraper.py                  # [TODO] Web scraper for The Batch
│   ├── chunking.py                 # [TODO] Text chunking logic
│   ├── embeddings.py               # [TODO] Text + Image embedding generation
│   ├── retriever.py                # [TODO] Multimodal retriever (from RAG/)
│   ├── llm_handler.py              # [TODO] LLM integration (Ollama/HF)
│   ├── database.py                 # [TODO] Vector DB operations (ChromaDB)
│   ├── tokenizing.py               # [TODO] Tokenization (from RAG/)
│   └── prompt.py                   # [TODO] Prompt templates (from RAG/)
├── app.py                          # [TODO] Gradio UI application (from RAG/)
├── evaluation/                     # [TODO] Evaluation scripts
│   ├── test_queries.json           # Test dataset
│   └── evaluate.py                 # Evaluation metrics
├── docs/                           # Documentation
│   └── technical_report.md         # [TODO] Technical approach report
├── tests/                          # Unit tests
├── env/                            # Virtual environment
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── CLAUDE.md                       # Claude Code instructions
└── todo.md                         # Implementation checklist
```

## Setup Instructions

### Prerequisites
- Python 3.9+
- Virtual environment (already created in `env/`)
- Ollama installed (or use HuggingFace models)

### Installation

1. Activate virtual environment:
```bash
source env/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required models (first run will auto-download):
   - sentence-transformers: `all-MiniLM-L6-v2`
   - CLIP: `openai/clip-vit-base-patch32`

4. Install Ollama (optional, for local LLM):
```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
```

### Usage

[TODO: Add usage instructions after implementation]

## Implementation Details

### Phase 2: Text Processing & Chunking

#### Approach
- **Method**: Recursive character text splitting with semantic awareness
- **Chunk size**: 400-600 tokens (~300-450 words) for optimal embedding quality
- **Overlap**: 50 tokens to maintain context across chunk boundaries
- **Separator priority**: `\n\n` (paragraphs) → `\n` (lines) → `. ` (sentences) → ` ` (words)

#### Chunking Logic
Split articles into semantically coherent chunks while preserving:
- Paragraph boundaries (primary)
- Sentence boundaries (secondary)
- Context through overlap

#### Metadata Preservation
Each chunk stores:
- `chunk_id`: Unique identifier (article_id + chunk_index)
- `article_id`: Parent article reference
- `article_title`: For context
- `article_url`: Source link
- `chunk_index`: Position in article
- `chunk_text`: The actual text content
- `word_count`: For statistics
- `associated_images`: Images near this chunk in original article

#### Image-Chunk Association
Images are distributed across chunks using a simple mathematical approach:
- **First chunk (chunk 0)**: Gets first 1-2 images (featured images)
- **Remaining chunks**: Remaining images distributed evenly across all other chunks
- **Limit**: Maximum 2 images per chunk to avoid overload

**Note**: This is a simplified approach that spreads images evenly by position, not by actual HTML proximity. A production system would track each image's position in the original HTML and associate it with the semantically closest chunk. For testing purposes, mathematical distribution is sufficient.

#### Output Format
Processed chunks saved to `data/processed/chunks.json`:
```json
[
  {
    "chunk_id": "article_1_chunk_0",
    "article_id": 1,
    "article_title": "...",
    "article_url": "...",
    "chunk_index": 0,
    "chunk_text": "...",
    "word_count": 450,
    "associated_images": ["article_1_img_0.jpg"]
  }
]
```

### Phase 1: Data Ingestion (Web Scraping)

#### Approach
- **Target**: The Batch articles from https://www.deeplearning.ai/the-batch/
- **Method**: BeautifulSoup4 for HTML parsing, requests for fetching
- **Pagination**: Iterate through `/the-batch/page/N/` to collect multiple articles
- **Rate limiting**: 1-2 second delays between requests to be respectful

#### Data Extraction
Each article contains:
- **Title**: `<h1>` or article heading
- **Date**: Publication date from metadata
- **URL**: Direct link to full article
- **Content**: Main article text (paragraphs)
- **Images**: Featured image + inline images with URLs and alt text
- **Tags**: Category tags if available

#### Storage Structure
```
data/raw/
  articles_batch_1.json      # First N articles
  articles_batch_2.json      # Next N articles (if needed)
data/images/
  article_123_img_0.jpg      # Images named by article ID
  article_123_img_1.jpg
```

#### Error Handling
- Skip articles with missing critical fields (title, content)
- Log failed image downloads but continue
- Save progress incrementally (batch by batch)

#### Current Limitations
**Note**: The current scraper implementation is not ideal for production use:

1. **URL Filtering**: Scrapes links from the homepage, but some lead to tag pages (`/the-batch/tag/...`) rather than actual articles. Tag pages lack proper article structure, so the scraper correctly skips them. Result: attempting 15 articles yielded only 3 valid articles.

2. **Duplicate Handling**: Re-running the scraper will create duplicates:
   - JSON files with the same `batch_name` get overwritten
   - Images with the same article_id get overwritten
   - No URL-based deduplication to detect already-scraped articles

3. **Update Detection**: No mechanism to detect when articles on the site have been updated and need re-scraping.

**For this project**: We proceed with 3 articles, which is sufficient for testing the full RAG pipeline (chunking, embeddings, retrieval, LLM integration).

**Production improvements needed**:
- Better URL filtering (sitemap, RSS feed, or link pattern matching)
- Duplicate detection using URL hashes or database tracking
- Update detection by comparing article modification dates or content hashes
- Incremental scraping that only fetches new/updated articles

## Development Progress

- [x] Phase 0: Setup & Planning
  - [x] README created
  - [x] Project structure folders
  - [x] requirements.txt
  - [x] .gitignore updates
  - [x] Dependencies installed
- [x] Phase 1: Data Ingestion
  - [x] Documented scraping approach
  - [x] Web scraper implementation
  - [x] Test with articles (3 articles scraped, ~5K words total)
- [x] Phase 2: Text Processing & Chunking
  - [x] Documented chunking strategy
  - [x] Chunking implementation (RecursiveCharacterTextSplitter)
  - [x] Image-chunk association logic
  - [x] Tested (71 chunks created, avg 66 words/chunk)
- [x] Phase 3: Text Embeddings & BM25
  - [x] Documented text retrieval approach
  - [x] Text embeddings implementation
  - [x] BM25 index setup
  - [x] Tested text retrieval (384-dim embeddings, hybrid search working)

### Phase 3: Text Embeddings & BM25

#### Approach
**Hybrid Retrieval Strategy**: Combine keyword-based (BM25) and semantic (embeddings) search for optimal text retrieval.

**Why Hybrid?**
- **BM25**: Excels at exact keyword matches (e.g., "Llama 3.2" finds exact model names)
- **Semantic**: Captures meaning (e.g., "AI safety" finds "alignment research", "responsible AI")
- **Combination**: Leverages both strengths for better recall and precision

**Text Embedding Model**
- **Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **Dimensions**: 384 (compact, fast)
- **Why**: Lightweight, runs on CPU, good semantic quality for general text

**BM25 Configuration**
- **Library**: rank-bm25 (fast, pure Python)
- **Tokenization**: NLTK word tokenizer
- **Parameters**: Default k1=1.5, b=0.75 (standard BM25 settings)

**Score Fusion**
- **Formula**: `final_score = α * BM25_normalized + (1-α) * semantic_cosine_similarity`
- **Default α**: 0.4 (60% semantic, 40% keyword)
- **Normalization**: Min-max scaling for BM25 scores to [0,1] range

**Caching Strategy**
- Text embeddings saved to `data/embeddings/text_embeddings.pkl` (reuse on reload)
- BM25 index saved to `data/cache/bm25_index.pkl`
- Tokenized docs saved to `data/cache/tokenized_docs.pkl`

## Evaluation

[TODO: Add evaluation results after implementation]

## License

[TODO: Add license if needed]

## Acknowledgments

- Based on existing RAG system patterns from `RAG/` folder
- The Batch articles from [DeepLearning.AI](https://www.deeplearning.ai/)