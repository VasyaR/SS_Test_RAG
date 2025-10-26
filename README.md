# Multimodal RAG System for The Batch

A Retrieval-Augmented Generation (RAG) system that retrieves relevant news articles from [The Batch](https://www.deeplearning.ai/the-batch/) using both textual and visual data. Users can query the system and receive AI-generated answers along with relevant articles and images.

**Current Dataset**: 18 articles (245 text chunks, 154 images) across 9 categories from The Batch.

---

## Table of Contents

1. [Quick Start (With Existing Data)](#quick-start-with-existing-data)
2. [Full Setup (Scraping from Scratch)](#full-setup-scraping-from-scratch)
3. [Project Overview](#project-overview)
4. [Technology Stack](#technology-stack)
5. [Project Structure](#project-structure)
6. [Architecture](#architecture)
7. [Limitations & Known Issues](#limitations--known-issues)
8. [Technical Implementation](#technical-implementation)

---

## Quick Start (With Existing Data)

**Prerequisites**: You already have the `data/` folder with scraped articles, embeddings, and Qdrant database.

### Step 1: Install Dependencies

```bash
# Create a python env in root dir (tested on linux)
python3 -m venv env

# Activate virtual environment
source env/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Step 2: Get Groq API Key

Get a free API key from [Groq Console](https://console.groq.com/keys) (14,400 requests/day free tier).

### Step 3: Launch the App

```bash
python3 -u app.py
```

Access the UI:
- **Local**: http://127.0.0.1:7860
- **Public URL**: Displayed in terminal (for WSL2/remote environments)

### Step 4: Use the Interface

1. **Setup tab** (first time only):
   - Enter your Groq API key
   - Click "Submit" to initialize the bot

2. **Article QA tab**:
   - Enter your question (e.g., "What are the latest AI model developments?")
   - (Optional) Filter by categories, date range
   - Toggle hybrid search (BM25 + Semantic)
   - View answer with sources and related images

---

## Full Setup (Scraping from Scratch)

**Use this guide if you don't have the `data/` folder** and need to scrape articles, generate embeddings, and build the database from scratch.

### Prerequisites

- Python 3.9+
- Virtual environment (create with `python3 -m venv env`)
- ~2GB disk space for data + models

### Step 1: Install Dependencies

```bash
# Activate virtual environment
source env/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

First run will auto-download models:
- `all-MiniLM-L6-v2` (text embeddings, ~90MB)
- `openai/clip-vit-base-patch32` (image embeddings, ~600MB)

### Step 2: Scrape Articles

```bash
cd src
python3 scraper.py --multi-category --per-category 2 --incremental
```

**CLI Arguments**:
- `--multi-category`: Scrape from all 9 category tabs
- `--per-category N`: Number of articles per category (default: 2)
- `--categories [list]`: Scrape specific categories only (e.g., `--categories "ML Research" "Business"`)
- `--incremental`: Skip already-scraped articles (useful for updates)

**Output**:
- Articles: `data/raw/articles_test_batch.json`
- Images: `data/images/article_*_img_*.{jpg,png}`

### Step 3: Chunk Articles

```bash
python3 chunking.py
```

**CLI Arguments** (optional):
- `--apply-clip`: Apply CLIP-based image-to-chunk matching (deprecated, not used in retrieval)
- `--threshold 0.5`: CLIP similarity threshold for matching

**Output**: `data/processed/chunks.json` (article text split into ~500 char chunks)

### Step 4: Generate Text Embeddings

```bash
python3 embeddings.py
```

**Output**:
- `data/embeddings/text_embeddings.pkl` (384-dim sentence transformer embeddings)
- `data/cache/bm25_index.pkl` (BM25 index for keyword search)
- `data/cache/tokenized_docs.pkl` (tokenized corpus)

### Step 5: Generate Image Embeddings

```bash
python3 image_embeddings.py
```

**Output**:
- `data/embeddings/image_embeddings.pkl` (512-dim CLIP embeddings)
- `data/embeddings/image_metadata.json` (image metadata with categories/dates)

### Step 6: Build Qdrant Database

```bash
python3 database.py
```

**Output**: `data/qdrant_db/` (vector database with text and image collections)

**Collections**:
- `text_chunks`: 245 text chunks with metadata (categories, dates, timestamps)
- `images`: 154 images with metadata for filtering

### Step 7: Launch the App

```bash
cd ..  # Return to project root
python3 -u app.py
```

Follow the [Quick Start Step 4](#step-4-use-the-interface) to use the interface.

---

## Project Overview

### Key Features

- **Hybrid text retrieval**: Combines BM25 (keyword) + semantic embeddings for optimal text search
- **Visual search**: CLIP-based image retrieval from query text
- **Smart image ranking**: First image from best-matched article, others from general CLIP search
- **Advanced filtering**: Filter by categories, date ranges, article IDs
- **LLM-powered answers**: Groq API (Llama 3.3 70B) generates natural language answers from retrieved context
- **Interactive UI**: Gradio interface with real-time query and filtering (first query is slow)

### Supported Categories

The system scrapes and filters across 9 categories from The Batch:
1. Weekly Issues
2. Andrew's Letters
3. Data Points
4. ML Research
5. Business
6. Science
7. Culture
8. Hardware
9. AI Careers

---

## Technology Stack

### Core Components

| Component | Technology | Why? |
|-----------|------------|------|
| **Web Scraping** | BeautifulSoup4 + Requests | Simple, reliable scraping for static content |
| **Text Embeddings** | sentence-transformers (`all-MiniLM-L6-v2`) | Lightweight (384-dim), fast on CPU, good semantic quality |
| **Keyword Search** | rank-bm25 | Complements semantic search, catches exact matches |
| **Image Embeddings** | CLIP (`openai/clip-vit-base-patch32`) | Multimodal embeddings (512-dim), text-to-image search |
| **Vector Database** | Qdrant (local mode) | Local, fast, advanced metadata filtering, no server needed |
| **LLM** | Groq API (Llama 3.3 70B) | Free tier (14,400 req/day), fast inference (500+ tok/s) |
| **LLM Library** | litellm | Unified API across providers (easy to switch) |
| **Text Chunking** | LangChain RecursiveCharacterTextSplitter | Semantic coherence, paragraph-aware splitting |
| **UI** | Gradio 5.9.1 | Fast prototyping, supports galleries, filters |

### Why Local + Cloud Hybrid?

- **Local** (embeddings, BM25, Qdrant): Free, private, fast for retrieval
- **Cloud** (Groq LLM): Free tier sufficient, faster than local LLM inference

---

## Project Structure

```
├── data/                              # Data storage (gitignored)
│   ├── raw/
│   │   └── articles_test_batch.json   # Scraped articles with metadata
│   ├── images/                        # Downloaded article images
│   │   └── article_*_img_*.{jpg,png}
│   ├── processed/
│   │   └── chunks.json                # Chunked article text
│   ├── embeddings/
│   │   ├── text_embeddings.pkl        # 384-dim text embeddings
│   │   ├── image_embeddings.pkl       # 512-dim CLIP embeddings
│   │   └── image_metadata.json        # Image metadata (categories, dates)
│   ├── cache/
│   │   ├── bm25_index.pkl             # BM25 keyword index
│   │   └── tokenized_docs.pkl         # Tokenized corpus for BM25
│   └── qdrant_db/                     # Qdrant vector database
│       ├── collection/text_chunks/
│       └── collection/images/
├── src/                               # Source code
│   ├── scraper.py                     # Web scraper (multi-category support)
│   ├── chunking.py                    # Text chunking (RecursiveCharacterTextSplitter)
│   ├── embeddings.py                  # Text embeddings + BM25 indexing
│   ├── image_embeddings.py            # CLIP image embeddings
│   ├── retriever.py                   # Multimodal retrieval (hybrid + CLIP)
│   ├── database.py                    # Qdrant database operations
│   ├── LLM_usage.py                   # ArticleQABot (Groq integration)
│   ├── prompt.py                      # System prompt template
│   └── __init__.py                    # Package initialization
├── app.py                             # Gradio UI (Setup + Article QA tabs)
├── env/                               # Virtual environment
├── requirements.txt                   # Python dependencies
├── .env.example                       # Template for API key configuration
├── .env                               # Your API key (gitignored)
├── README.md                          # This file
├── CLAUDE.md                          # Claude Code instructions
├── todo.md                            # Development history & progress
└── session{n}.txt                       # Claude Code chat history
```

**Note on session files**: `session4.txt` and `session5.txt` contain the full conversation history with Claude Code. Useful for understanding implementation decisions and debugging history.

---

## Architecture

### End-to-End Query Flow

```
User Query (text)
    ↓
┌─────────────────────────────────────┐
│  1. Query Processing                │
│  - Tokenize for BM25                │
│  - Embed with sentence-transformers │
│  - Embed with CLIP (for images)     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  2. Hybrid Text Retrieval           │
│  - BM25 scores (keyword matching)   │
│  - Semantic scores (cosine sim)     │
│  - Fuse: 0.4*BM25 + 0.6*Semantic    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  3. Smart Image Retrieval           │
│  - Best CLIP image from top article │
│  - Fill to 3 with general CLIP      │
│  - Avoid duplicates                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  4. Metadata Filtering              │
│  - Categories (array match)         │
│  - Date ranges (Unix timestamps)    │
│  - Article IDs                      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  5. LLM Answer Generation           │
│  - Build context from top K chunks  │
│  - Format: [Title] (Date)\nContent  │
│  - Groq API (Llama 3.3 70B)         │
└─────────────────────────────────────┘
    ↓
Answer + Sources + Images
    ↓
Gradio UI Display
```

### Retrieval Parameters

- **Text retrieval**: Top 5 chunks (default), hybrid BM25+semantic with α=0.4
- **Image retrieval**: Up to 3 images (1 from top article, 2 general)
- **Context window**: ~2000 tokens for LLM
- **Filters**: Optional categories, date ranges

---

## Limitations & Known Issues

### 1. Qdrant Local Mode (No Concurrent Access)

**Current Setup**: Uses Qdrant in local/embedded mode (`QdrantClient(path=...)`), which provides file-based persistence without requiring a server.

**Limitation**:
- ✗ **Cannot scrape new articles while app is running** - Qdrant local mode uses file locking and only allows one client at a time
- ✗ **Must stop app to update database** - To add new articles, you must stop the Gradio app, run the full pipeline (scrape → chunk → embed → rebuild database), then restart

**Workflow for Adding New Articles**:
1. Stop the running Gradio app (`Ctrl+C` or `pkill python3`)
2. Scrape new articles: `cd src && python3 scraper.py --multi-category --per-category 1 --incremental`
3. Process chunks: `python3 chunking.py`
4. Regenerate embeddings: `python3 embeddings.py` and `python3 image_embeddings.py`
5. Rebuild Qdrant: `python3 database.py`
6. Restart the app: `cd .. && python3 -u app.py`

**Production Solution**:

For systems requiring continuous scraping while serving queries, switch to **Qdrant Server Mode**:

```bash
# Run Qdrant as a separate server (supports concurrent access)
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# Update src/database.py:
client = QdrantClient(host="localhost", port=6333)  # instead of path=...
```

### 2. Image-Chunk Association (Deprecated Feature)

**Historical Background**: Early development (Phase 2) implemented CLIP-based semantic matching to link specific images to text chunks, stored in the `associated_images` field.

**Current Status**: **This feature is no longer used** in the retrieval system (deprecated).

**Why Deprecated?**
- Smart image retrieval (best from top article + general CLIP) proved more effective
- Not restricted to pre-linked images, can surface any relevant image
- Simpler implementation, less maintenance

**What Remains**:
- `associated_images` field still exists in `data/processed/chunks.json` for historical/debugging purposes
- Data is accurate (CLIP similarity threshold 0.5) but not actively used in `src/retriever.py`

**Future**: May be removed entirely in a database cleanup phase.

### 3. Content Update Detection

**Limitation**: The scraper detects new articles by URL only. If an existing article is edited on The Batch website, the scraper will not detect or update the changes.

**Workaround**:
1. Manually locate the article in `data/raw/articles_test_batch.json`
2. Delete that article entry
3. Re-run scraper: `python3 scraper.py --multi-category --incremental`

### 4. Gradio Gallery Bug (Fixed)

**Issue**: Gradio 5.9.1 had a bug in Gallery component API schema generation causing `TypeError: argument of type 'bool' is not iterable`.

**Fix**: Pinned `pydantic==2.10.6` in `requirements.txt` (workaround until Gradio fixes upstream).

**Reference**: [Gradio GitHub Issue #11084](https://github.com/gradio-app/gradio/issues/11084)

---

## Technical Implementation

### Phase 1: Data Ingestion (Web Scraping)

**Approach**: BeautifulSoup4 for HTML parsing, requests for fetching

**Features**:
- Multi-category scraping across 9 categories
- Date extraction via JSON-LD, time tags, meta tags (fallback chain)
- Incremental scraping with URL-based deduplication
- Image handling: RGBA/P images saved as PNG, filters data URIs
- Rate limiting: 1-2 second delays between requests

**Output**: `data/raw/articles_test_batch.json` + images in `data/images/`

---

### Phase 2: Text Processing & Chunking

**Approach**: RecursiveCharacterTextSplitter with semantic awareness

**Configuration**:
- **Chunk size**: ~500 characters (~100-125 tokens)
- **Overlap**: 50 characters for context continuity
- **Separator priority**: Paragraphs → Lines → Sentences → Words
- **Title inclusion**: Article title prepended to each chunk for better search

**Metadata per Chunk**:
- `chunk_id`, `article_id`, `article_title`, `article_url`
- `article_date`, `article_timestamp`, `article_categories`
- `chunk_index`, `total_chunks`, `word_count`, `chunk_text`

**Output**: `data/processed/chunks.json` (245 chunks from 18 articles)

---

### Phase 3: Text Embeddings & BM25

**Hybrid Retrieval Strategy**: Combine keyword-based (BM25) and semantic (embeddings) search

**Why Hybrid?**
- **BM25**: Excels at exact keyword matches (e.g., "Llama 3.2")
- **Semantic**: Captures meaning (e.g., "AI safety" finds "alignment research")

**Text Embedding Model**: `all-MiniLM-L6-v2`
- 384 dimensions, normalized for cosine similarity
- Fast on CPU, good semantic quality

**BM25 Configuration**: rank-bm25 library
- Tokenization: NLTK word tokenizer
- Parameters: k1=1.5, b=0.75 (standard BM25)

**Score Fusion**: `final_score = 0.4 * BM25_normalized + 0.6 * semantic_score`

**Output**:
- `data/embeddings/text_embeddings.pkl`
- `data/cache/bm25_index.pkl`
- `data/cache/tokenized_docs.pkl`

---

### Phase 4: Image Embeddings with CLIP

**Approach**: CLIP for visual-semantic embeddings (images and text in same vector space)

**CLIP Model**: `openai/clip-vit-base-patch32`
- 512 dimensions, normalized for cosine similarity
- Trained on 400M image-text pairs
- Zero-shot text-to-image retrieval

**Image Processing**:
- Preprocessing: CLIP's built-in transforms (resize to 224x224, normalize)
- Error handling: Skip corrupted/missing images gracefully

**Metadata Enrichment** :
- Loads article metadata from `articles_test_batch.json`
- Includes: `article_categories`, `article_date`, `article_timestamp` for filtering

**Output**:
- `data/embeddings/image_embeddings.pkl` (154 images)
- `data/embeddings/image_metadata.json` (with full metadata)

---

### Phase 5: Multimodal Vector Database

**Approach**: Qdrant with two separate collections for text and images

**Why Qdrant?**
- Local/fast (Rust-powered), no API costs
- Advanced metadata filtering (arrays, date ranges, complex queries)
- Persistent storage in `data/qdrant_db/`

**Collections**:

1. **`text_chunks`**: 245 text embeddings
   - Dimensions: 384 (sentence-transformers)
   - Metadata: `article_id`, `article_title`, `article_url`, `article_date`, `article_timestamp`, `article_categories` (array), `chunk_id`, `chunk_index`, `word_count`
   - Payload: Full chunk text

2. **`images`**: 154 image embeddings
   - Dimensions: 512 (CLIP)
   - Metadata: `article_id`, `article_title`, `article_url`, `article_date`, `article_timestamp`, `article_categories` (array), `image_path`, `full_path`

**Metadata Filtering Examples**:

```python
# Filter by multiple categories
Filter(must=[
    FieldCondition(key="article_categories", match=MatchAny(any=["ML Research", "Business"]))
])

# Filter by date range (October 2025)
Filter(must=[
    FieldCondition(key="article_timestamp", range=Range(gte=1727740800, lt=1730419200))
])
```

---

### Phase 6: Multimodal Retrieval System

**Hybrid Text Retrieval**:
- BM25 scores (keyword) + Semantic scores (cosine similarity)
- Formula: `text_score = 0.4 * BM25_normalized + 0.6 * semantic_score`
- Retrieves top K chunks (default: 5)

**Smart Image Retrieval** (Phase 7 enhancement):
1. **First image**: Best CLIP similarity from best text-matched article
2. **Remaining images**: General CLIP search across all articles (avoid duplicates)
3. **Total**: Up to 3 images per query
4. **Respects filters**: Category/date filters apply to both text and images

**Retrieval Modes**:
- Text-only (hybrid BM25 + semantic)
- Text-to-image (CLIP text encoder)
- Multimodal (combines text + image scores)
- Image-to-image (CLIP image encoder, for future feature)

---

### Phase 7: LLM Integration

**Question-Answering System**: Groq API with Llama 3.3 70B Versatile

**Components**:

1. **ArticleQABot** (`src/LLM_usage.py`):
   - Initializes with MultimodalRetriever
   - Retrieves top K chunks + images
   - Builds context with article titles and dates
   - Generates answer via Groq API

2. **Prompt Template** (`src/prompt.py`):
   - System role: "You are a helpful AI news assistant..."
   - Instructions: Answer from retrieved articles only, cite sources, admit uncertainty

3. **Gradio UI** (`app.py`):
   - **Setup tab**: Enter Groq API key (one-time initialization)
   - **Article QA tab**: Query input, filters (categories, date range), result count slider, hybrid search toggle

**API Configuration**:
- Library: litellm (unified API for multiple providers)
- Model: `groq/llama-3.3-70b-versatile`
- Rate limits: 14,400 requests/day (free tier)
- Context window: ~2000 tokens from retrieved chunks

**Response Format**:
```python
{
    "answer": "Generated answer text...",
    "sources": [
        {"title": "...", "url": "...", "date": "...", "categories": [...]}
    ],
    "images": [
        {"image_path": "...", "article_id": ..., "metadata": {...}}
    ]
}
```