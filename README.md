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
- **Qdrant**: Store and query embeddings with metadata filtering
  - **Why**: Local, supports metadata filtering, easy to use, no server needed

#### LLM
- **Groq API** (Llama 3.3 70B Versatile): Cloud-based LLM via free API
  - **Why**: Production-ready, faster than local (500+ tokens/sec), free tier (14,400 requests/day), same as RAG/ architecture
  - **Library**: litellm for unified API access
  - **Model**: `groq/llama-3.3-70b-versatile`
  - **Alternative**: Can switch to other providers (OpenAI, Anthropic) via litellm without code changes

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
- **Method**: Paragraph-based with 500 character chunks (~100-125 tokens)
- **Why**: Articles are well-structured with clear paragraphs
- **Overlap**: 50 characters to maintain context across chunks
- **Title inclusion**: Article title prepended to each chunk for better BM25/semantic search
- **Image linking**: CLIP-based semantic matching
  - Images matched to chunks via CLIP vision-language similarity
  - Normalized similarity threshold: 0.5 (balanced coverage vs precision)
  - Result: ~50% images assigned, ~97% chunks have images
  - Unassigned images are typically decorative/generic with low semantic relevance

### Multimodal Fusion
- **Text score**: `α * BM25_score + (1-α) * Semantic_score`
- **Image score**: CLIP similarity between query and image embeddings
- **Final score**: `β * Text_score + (1-β) * Image_score`
- **Parameters**: User-adjustable in UI (like existing RAG/ system)

### Metadata Filtering
- Stored metadata: `date`, `title`, `article_id`, `url`, `image_path`, `tags`
- Qdrant filters for metadata-based filtering (categories, date ranges, article IDs)
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
│   ├── database.py                 # Vector DB operations (Qdrant)
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

#### Running the Gradio App

1. **Activate environment**:
```bash
source env/bin/activate
```

2. **Launch the app**:
```bash
python3 -u app.py
```

3. **Access the UI**:
   - Local: http://127.0.0.1:7860
   - Public (share link): Displayed in terminal output

4. **Setup tab** (first time):
   - Enter your Groq API key (get free key at https://console.groq.com/keys)
   - Click "Submit" to initialize the bot

5. **Article QA tab**:
   - Enter your question
   - (Optional) Select categories to filter
   - (Optional) Set date range (format: YYYY-MM-DD)
   - Adjust number of results (1-10)
   - Toggle hybrid search (BM25 + Semantic)
   - Click "Ask Question"
   - View answer, sources, and related image

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

#### Phase 6.5 Enhancements (Implemented)

**Enhanced Metadata Extraction**:
- ✅ Date extraction via JSON-LD, time tags, meta tags (fallback chain)
- ✅ Multi-category support (articles can belong to multiple categories)
- ✅ Removed author field (not used by The Batch)

**Multi-Category Scraping**:
- ✅ Scrape from 9 category tabs: Weekly Issues, Andrew's Letters, Data Points, ML Research, Business, Science, Culture, Hardware, AI Careers
- ✅ Balanced distribution: specify articles per category
- ✅ CLI support: `python scraper.py --multi-category --per-category 2`

**Image Handling**:
- ✅ RGBA/P images saved as PNG (not JPEG)
- ✅ data: URI placeholders filtered out
- ✅ Extension tracking in metadata

**Incremental Scraping**:
- ✅ URL-based deduplication
- ✅ Skip existing articles automatically
- ✅ Pagination support (multiple pages)

**Current Dataset**: 18 articles across all 9 categories with proper dates and metadata.

#### Database Rebuild (In Progress)

After Phase 6.5 optimizations, the database needs to be rebuilt with the new data structure:

**What's Changed**:
- 245 new chunks (up from old dataset) with article titles prepended
- Enhanced metadata: dates (JSON-LD extracted), multi-category tags
- CLIP-matched images (78 images assigned to chunks at threshold 0.5)
- All 154 images available for embedding (not just the 78 assigned ones)

**Rebuild Steps (Phases 3-5)**:
1. **Phase 3**: Regenerate text embeddings and BM25 index from 245 chunks with titles
2. **Phase 4**: Regenerate image embeddings for all 154 images (CLIP ViT-B/32)
3. **Phase 5**: Rebuild Qdrant with updated chunks, embeddings, and metadata

**Commands**:
```bash
# Phase 3: Text embeddings & BM25
cd src
source ../env/bin/activate
python embeddings.py

# Phase 4: Image embeddings
python image_embeddings.py

# Phase 5: Qdrant
python database.py
```

#### Remaining Limitations

**Content-based update detection**: No mechanism to detect when already-scraped articles have been edited (only detects new articles by URL).

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
- [x] Phase 4: Image Embeddings with CLIP
  - [x] Documented image embedding approach
  - [x] CLIP implementation
  - [x] Generated image embeddings
  - [x] Tested image retrieval (10 images, 512-dim embeddings, text-to-image working)
- [x] Phase 5: Multimodal Vector Database
  - [x] Documented Qdrant approach
  - [x] Implemented database module
  - [x] Loaded text and image embeddings
  - [x] Tested metadata filtering (71 text chunks, 10 images stored)
- [x] Phase 6: Multimodal Retrieval System
  - [x] Documented retrieval strategy
  - [x] Hybrid text retrieval implementation
  - [x] Image retrieval implementation
  - [x] Multimodal fusion implementation
  - [x] Tested end-to-end retrieval (hybrid scores, metadata filtering working)
- [x] Phase 6.5: System Optimizations
  - [x] Enhanced metadata extraction (date from JSON-LD, multi-category support)
  - [x] CLIP-based semantic image-chunk assignment (threshold 0.5)
  - [x] Incremental scraping with deduplication
  - [x] Multi-category scraping (9 categories)
  - [x] Article titles in chunks for better search
  - [x] CLI interfaces for scraper and chunker
  - [x] Migrated from ChromaDB to Qdrant
  - [x] Added Unix timestamp support for date range filtering
  - [x] Categories stored as arrays for multi-category filtering
  - [x] Rebuilt database with 245 chunks and 154 images
  - [x] All advanced filtering tested and working
- [x] Phase 7: LLM Integration
  - [x] Created prompt template for article Q&A assistant
  - [x] Implemented ArticleQABot with Groq API (Llama 3.3 70B)
  - [x] Built Gradio UI with Setup + Article QA tabs
  - [x] Added metadata filtering UI (categories, date ranges)
  - [x] Implemented smart image retrieval (best from top article + general CLIP)
  - [x] Fixed image metadata (added categories/dates/timestamps)
  - [x] Rebuilt Qdrant with enriched image metadata
  - [x] Fixed Gradio Gallery bug (pydantic 2.10.6)
  - [x] End-to-end tested and working

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

### Phase 4: Image Embeddings with CLIP

#### Approach
**Visual-Semantic Embeddings**: Use CLIP to generate embeddings that understand both images and text in the same vector space.

**Why CLIP?**
- **Multimodal**: Trained on 400M image-text pairs, understands visual concepts
- **Zero-shot**: Can match images to text queries without fine-tuning
- **Same embedding space**: Text and image embeddings are directly comparable
- **Free/Local**: OpenAI's CLIP models are open-source

**CLIP Model**
- **Model**: `openai/clip-vit-base-patch32`
- **Dimensions**: 512 (shared text/image space)
- **Why**: Good balance of speed and quality, runs on CPU

**Image Processing**
- **Preprocessing**: CLIP's built-in transforms (resize, normalize)
- **Input size**: 224x224 pixels (CLIP standard)
- **Error handling**: Skip corrupted/missing images gracefully

**Image-Text Query Support**
- **Text query**: Encode query text with CLIP text encoder → compare to image embeddings
- **Image query**: Encode query image with CLIP image encoder → find similar images
- **Both supported** in the same embedding space

**Caching Strategy**
- Image embeddings saved to `data/embeddings/image_embeddings.pkl`
- Image metadata (paths, article_ids) saved to `data/embeddings/image_metadata.json`
- Reuse on reload (only embed new images)

### Phase 5: Multimodal Vector Database

#### Approach
**Unified Storage with Qdrant**: Store text and image embeddings with advanced metadata filtering.

**Why Qdrant?**
- **Local/Fast**: Rust-powered, runs locally with no API costs
- **Advanced filtering**: Native support for arrays, date ranges, complex queries
- **Multi-collection**: Separate collections for text chunks and images
- **Vector similarity**: Cosine distance search
- **Production-ready**: High performance, scales to millions of vectors

**Database Structure**
Two separate collections:
1. **`text_chunks`**: Text embeddings with metadata
   - Embeddings: 384-dim (sentence-transformers), cosine distance
   - Metadata: `article_id`, `article_title`, `article_url`, `article_date`, `article_timestamp`, `article_categories` (array), `chunk_id`, `chunk_index`, `word_count`
   - Payload: Full chunk text for context

2. **`images`**: Image embeddings with metadata
   - Embeddings: 512-dim (CLIP), cosine distance
   - Metadata: `article_id`, `article_title`, `image_path`, `full_path`
   - Payload: Image file paths

**Metadata Filtering Examples**
```python
from qdrant_client.models import Filter, FieldCondition, MatchAny, Range

# Filter by multiple categories
results = client.search(
    collection_name="text_chunks",
    query_vector=query_emb,
    query_filter=Filter(must=[
        FieldCondition(key="article_categories", match=MatchAny(any=["ML Research", "Business"]))
    ]),
    limit=5
)

# Filter by date range (October 2025)
results = client.search(
    collection_name="text_chunks",
    query_vector=query_emb,
    query_filter=Filter(must=[
        FieldCondition(key="article_timestamp", range=Range(gte=1727740800, lt=1730419200))
    ]),
    limit=5
)

# Filter by article ID
results = client.search(
    collection_name="text_chunks",
    query_vector=query_emb,
    query_filter=Filter(must=[
        FieldCondition(key="article_id", match=MatchAny(any=[0, 1]))
    ]),
    limit=5
)
```

**Key Features**:
- **Array support**: Categories stored as arrays, filter by multiple categories
- **Date ranges**: Unix timestamps enable proper date range queries
- **Persistent storage**: Data stored in `data/qdrant_db/` directory
- **Separate collections**: Different dimensions (384 vs 512), independent querying

### Phase 6: Multimodal Retrieval System

#### Approach
**Unified Multimodal Retrieval**: Combine BM25, semantic text search, and CLIP image search into a single retrieval system with score fusion.

**Retrieval Pipeline**
1. **Text Query Processing**:
   - Tokenize query → BM25 scores
   - Embed query → Semantic scores
   - Normalize and fuse scores
2. **Image Query Processing** (optional):
   - Embed query text with CLIP → Image similarity scores
   - Or embed query image → Find similar images
3. **Score Fusion**:
   - Combine text and image scores
   - Return unified ranked results

**Hybrid Text Retrieval**
- **Formula**: `text_score = α * BM25_normalized + (1-α) * semantic_score`
- **α parameter**: Default 0.4 (60% semantic, 40% BM25)
- **BM25 normalization**: Min-max scaling to [0,1] range
- **Semantic scores**: Already in [0,1] (cosine similarity with normalized embeddings)

**Multimodal Fusion**
- **Formula**: `final_score = β * text_score + (1-β) * image_score`
- **β parameter**: Default 0.7 (70% text, 30% image)
- **Use case**: User provides text query, system retrieves both relevant text chunks AND images

**Retrieval Modes**
1. **Text-only**: Uses hybrid BM25 + semantic (default)
2. **Text-to-image**: Uses CLIP text encoder to find relevant images
3. **Multimodal**: Combines text and image retrieval scores
4. **Image-to-image**: Uses CLIP image encoder (for future image upload feature)

**Metadata Filtering Support**
- Filter by `article_id`: Get results from specific articles
- Filter by `word_count`: Find longer/shorter chunks
- Combine with score-based ranking

**Output Format**
```python
{
    "text_results": [
        {
            "chunk_id": "article_1_chunk_0",
            "text": "...",
            "score": 0.85,
            "metadata": {...}
        }
    ],
    "image_results": [
        {
            "image_path": "article_1_img_0.jpg",
            "score": 0.72,
            "metadata": {...}
        }
    ]
}
```

### Phase 7: LLM Integration ✅ COMPLETED

#### Approach
**Question-Answering System**: Use Groq API with Llama 3.3 70B to generate natural language answers from retrieved article context.

**LLM Architecture** (following RAG/ pattern)
1. **ArticleQABot Class** (`src/LLM_usage.py`):
   - Initialize with MultimodalRetriever
   - System prompt for article Q&A assistant
   - Answer generation with retrieved context

2. **Prompt Template** (`src/prompt.py`):
   - System role: "You are a helpful AI news assistant..."
   - Instructions: Answer from retrieved articles, cite sources, admit when uncertain
   - Format: Concise answers with article references

3. **Context Building**:
   - Retrieve top K chunks (default: 5)
   - Format: `[Article Title] (Date)\nContent...\n---`
   - Include article URLs for citations
   - Limit context to ~2000 tokens to fit LLM window

**API Configuration**
- **Library**: litellm (unified API for multiple providers)
- **Model**: `groq/llama-3.3-70b-versatile`
- **API Key**: User-provided via Gradio UI (Setup tab)
- **Rate Limits**: 14,400 requests/day (free tier)
- **Fallback**: Can switch to other models via litellm config

**Smart Image Retrieval Strategy**
1. **First image**: Best CLIP similarity from best text-matched article
2. **Remaining images**: General CLIP search (avoid duplicates)
3. **Total**: Up to 3 images per query
4. **Filters**: Respects category/date filters on both text and images

**Response Format**
```python
{
    "answer": "Generated answer text...",
    "context": "Retrieved article chunks...",
    "sources": [
        {"title": "...", "url": "...", "date": "...", "categories": [...]}
    ],
    "images": [
        {"image_path": "...", "article_id": ..., "metadata": {...}}
    ]
}
```

**Metadata Filtering Integration**
- UI provides category/date filters → build_filter()
- Filter passed to retriever → filtered results → LLM context
- Example: "What's new in transformers?" + category="ML Research" + date_range="Oct 2025"

**Gradio UI** (`app.py`)
- **Tab 1 (Setup)**: Enter Groq API key (get one at https://console.groq.com/keys)
- **Tab 2 (Article QA)**:
  - Query input with optional filters (categories, date range, result count)
  - Hybrid search toggle (BM25 + Semantic)
  - Answer display with sources and related images gallery

**Bug Fixes**
- ✅ Fixed Gradio 5.9.1 Gallery component error by pinning pydantic==2.10.6
- ✅ Added full metadata (categories, dates, timestamps) to image collection for filtering

## Limitations & Known Issues

### Qdrant Local Mode (Embedded)
**Current Setup**: Uses Qdrant in local/embedded mode (`QdrantClient(path=...)`), which provides file-based persistence without requiring a server.

**Limitation - No Concurrent Access**:
- ✗ **Cannot scrape new articles while app is running** - Qdrant local mode uses file locking and only allows one client at a time
- ✗ **Must stop app to update database** - To add new scraped articles, you must stop the Gradio app, regenerate embeddings, rebuild Qdrant collections, then restart the app

**Workflow for Adding New Articles**:
1. Stop the running Gradio app (kill the process)
2. Scrape new articles: `python3 src/scraper.py --multi-category --per-category 1`
3. Process chunks: `python3 src/chunking.py`
4. Regenerate embeddings: `python3 src/embeddings.py` and `python3 src/image_embeddings.py`
5. Rebuild Qdrant: `python3 src/database.py`
6. Restart the app: `python3 -u app.py`

**Production Solution**:
For a system that needs continuous scraping while serving queries, switch to **Qdrant Server Mode**:
```bash
# Run Qdrant as a separate server (supports concurrent access)
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# Update code to use server mode
client = QdrantClient(host="localhost", port=6333)  # instead of path=...
```

### Image-Chunk Association (Deprecated)
**Historical Feature**: Phase 2 implemented CLIP-based semantic matching to link specific images to text chunks (`associated_images` field in chunks).

**Current Status**: **This feature is no longer used in retrieval** (deprecated as of Phase 7).

**Why Deprecated?**:
- **Simpler approach**: Smart image retrieval (best from top article + general CLIP) proved more effective
- **Better coverage**: Not restricted to pre-linked images, can surface any relevant image
- **Less complexity**: Removed dependency on chunk-image assignments

**What Remains**:
- `associated_images` field still exists in `chunks.json` for historical/debugging purposes
- Data in the field is accurate (CLIP similarity threshold 0.5) but not actively used

**Future**: The field may be removed entirely in a database cleanup phase.

### Content Update Detection
**Limitation**: The scraper only detects new articles by URL. If an existing article is edited on The Batch website, the scraper will not detect or update the changes.

**Workaround**: Manually delete the article from `data/raw/articles_test_batch.json` and re-scrape to get the updated version.

## Evaluation

[TODO: Add evaluation results after implementation]

## License

[TODO: Add license if needed]

## Acknowledgments

- Based on existing RAG system patterns from `RAG/` folder
- The Batch articles from [DeepLearning.AI](https://www.deeplearning.ai/)