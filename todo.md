# Multimodal RAG System - Implementation Plan

## Project Overview
Building a Multimodal RAG system for The Batch articles with text + image retrieval, using free/local models and Gradio UI.

## Workflow for Each Step
1. **Document** approach in README.md (brief: what, why, how)
2. **Implement** the component
3. **Test** functionality
4. **Update** project structure in README.md
5. **Move to next step**

---

## Todo List

### Phase 0: Setup & Planning
- [x] Create initial README.md with project overview
- [x] Set up project structure (folders)
- [x] Create requirements.txt with dependencies
- [x] Document technology stack choices in README

### Phase 1: Data Ingestion (The Batch Scraper)
- [x] **Document**: Web scraping approach in README (BeautifulSoup, structure, pagination)
- [x] Implement web scraper for The Batch articles
- [x] Extract: title, date, text content, images URLs, article URL, categories
- [x] Download and save images locally (RGBA→PNG conversion)
- [x] Save raw articles as JSON
- [x] **Test**: Scraped 18 articles across 9 categories
- [x] **Update** project structure in README

### Phase 2: Text Processing & Chunking
- [x] **Document**: Chunking strategy in README (paragraph-based, 500 chars)
- [x] Implement text chunking logic (langchain RecursiveCharacterTextSplitter)
- [x] Link chunks to their parent articles
- [x] Link chunks to associated images via CLIP semantic matching
- [x] Prepend article title to chunks for better search
- [x] **Test**: 245 chunks created with metadata
- [x] **Update** project structure in README

### Phase 3: Text Embeddings & BM25
- [x] **Document**: Text retrieval approach (BM25 + sentence-transformers, hybrid search)
- [x] Set up sentence-transformers (all-MiniLM-L6-v2)
- [x] Generate embeddings for text chunks (384-dim, normalized)
- [x] Set up BM25 index with NLTK tokenization
- [x] **Test**: Text retrieval working with hybrid search
- [x] **Update** project structure in README

### Phase 4: Image Embeddings (CLIP)
- [x] **Document**: Image embedding approach (CLIP ViT-B/32)
- [x] Set up CLIP model (openai/clip-vit-base-patch32)
- [x] Generate embeddings for ALL 154 images (512-dim)
- [x] Save embeddings as pickle files
- [x] **Test**: Text-to-image and image-to-image retrieval working
- [x] **Update** project structure in README

### Phase 5: Multimodal Vector Database
- [x] **Document**: Qdrant database approach (replaced ChromaDB)
- [x] Set up Qdrant with separate collections for text and images
- [x] Store metadata: article_id, title, url, date, timestamp, categories (array)
- [x] Implement advanced filtering: multi-category, date ranges
- [x] **Test**: All filtering working (categories, dates, article IDs)
- [x] **Update** project structure in README

### Phase 6: Multimodal Retrieval System
- [x] **Document**: Fusion strategy (text + image scores, weighting, ranking)
- [x] Implement multimodal retriever
- [x] Combine text scores + image scores (weighted fusion)
- [x] Add metadata filtering integration
- [x] Return: top K chunks + images + article metadata
- [x] **Test**: End-to-end retrieval tested
- [x] **Update** project structure in README

### Phase 6.5: System Optimizations
- [x] Enhanced metadata extraction (JSON-LD dates, multi-category support)
- [x] CLIP-based semantic image-chunk assignment (threshold 0.5)
- [x] Incremental scraping with deduplication
- [x] Multi-category scraping (9 categories: Weekly Issues, ML Research, etc.)
- [x] Article titles in chunks for better BM25/semantic search
- [x] CLI interfaces for scraper and chunker
- [x] Migrated from ChromaDB to Qdrant
- [x] Unix timestamps for date range filtering
- [x] Categories as arrays for multi-category filtering
- [x] Database rebuilt: 245 chunks, 154 images

### Phase 7: LLM Integration ✅ COMPLETED
- [x] **Document**: LLM choice (Groq API chosen over Ollama for speed/free tier)
- [x] Set up Groq API with Llama 3.3 70B via litellm
- [x] Create prompt template for article Q&A (src/prompt.py)
- [x] Implement ArticleQABot with retrieved context (src/LLM_usage.py)
- [x] Handle context window (~2000 tokens from top K chunks)
- [x] **Test**: Ask questions, verify answer quality - working!
- [x] **Update** project structure in README

#### Phase 7 Enhancements
- [x] Smart image retrieval: Best CLIP from top article + general search
- [x] Fixed image metadata: Added categories, dates, timestamps
- [x] Rebuilt Qdrant database with enriched image metadata
- [x] Fixed Gradio 5.9.1 Gallery bug (pinned pydantic==2.10.6)

### Phase 8: Gradio UI ✅ COMPLETED
- [x] **Document**: UI design (Setup + Article QA tabs)
- [x] Set up Gradio interface (app.py with 2 tabs)
- [x] Input: Query textbox with optional filters (categories, date range)
- [x] Output 1: LLM-generated answer
- [x] Output 2: Retrieved article sources (title, date, URL, categories)
- [x] Output 3: Image gallery for retrieved images (up to 3)
- [x] Add retrieval mode selector (Hybrid BM25+Semantic toggle)
- [x] Add metadata filtering UI (categories checkboxes, date range inputs)
- [x] **Test**: Full user flow - query → answer + images + articles - WORKING!
- [x] **Update** project structure in README

### Phase 9: System Evaluation
- [ ] **Document**: Evaluation metrics and approach
- [ ] Create test query dataset (10-15 diverse queries)
- [ ] Implement evaluation metrics (retrieval precision, answer relevance)
- [ ] Run evaluation on test queries
- [ ] Document results in README
- [ ] **Update** project structure in README

### Phase 10: Documentation & Cleanup
- [ ] Write comprehensive setup instructions in README
- [ ] Add usage examples and screenshots to README
- [ ] Create technical report (doc/ folder)
- [ ] Clean up code (remove debug prints, add docstrings)
- [ ] Verify all files follow CLAUDE.md conventions
- [ ] Final project structure update in README

### Phase 11: Demo & Delivery
- [ ] Record demo video
- [ ] Test full setup from scratch (verify README instructions)
- [ ] Create submission package
- [ ] (Optional) Cloud deployment setup

---

## Technology Stack (Final Implementation)

### Core Components
- **Web Scraping**: BeautifulSoup4 + requests
- **Text Embeddings**: sentence-transformers (all-MiniLM-L6-v2, 384-dim)
- **Image Embeddings**: CLIP (openai/clip-vit-base-patch32, 512-dim)
- **Vector Database**: Qdrant (local mode, advanced filtering with arrays & date ranges)
- **BM25**: rank_bm25 with NLTK tokenization
- **LLM**: Groq API (Llama 3.3 70B Versatile) via litellm ✅
- **UI**: Gradio 5.9.1 ✅
- **Chunking**: langchain RecursiveCharacterTextSplitter (500 chars, 50 overlap)

**Why Groq over Ollama?**
- Free tier: 14,400 requests/day
- Faster inference: 500+ tokens/sec (vs local LLM ~10-50 tok/s)
- No local GPU/RAM requirements
- Production-ready quality (Llama 3.3 70B)

### File Reuse from RAG/
- ✅ `retriever.py` - 70% reused (BM25 + semantic logic, adapted for Qdrant)
- ✅ `app.py` - 60% reused (Gradio structure, modified for multimodal)
- ✅ `LLM_usage.py` - 40% reused (Groq integration via litellm)
- ✅ `prompt.py` - 20% reused (changed prompt content for article Q&A)

---

## Project Structure (To be filled as we progress)
```
├── data/                    # Data storage
│   ├── raw/                 # Raw scraped articles (JSON)
│   ├── images/              # Downloaded images
│   ├── processed/           # Processed chunks
│   ├── embeddings/          # Saved embeddings
│   └── cache/               # BM25, tokenized docs cache
├── src/                     # Source code
│   ├── scraper.py           # Web scraper for The Batch
│   ├── chunking.py          # Text chunking logic
│   ├── embeddings.py        # Text + Image embedding generation
│   ├── retriever.py         # Multimodal retriever
│   ├── llm_handler.py       # LLM integration (Ollama/HF)
│   ├── database.py          # Vector DB operations
│   ├── tokenizing.py        # From RAG/ (reused)
│   └── prompt.py            # Prompt templates
├── app.py                   # Gradio UI application
├── evaluation/              # Evaluation scripts
│   ├── test_queries.json    # Test dataset
│   └── evaluate.py          # Evaluation metrics
├── docs/                    # Documentation
│   └── technical_report.md  # Technical approach report
├── tests/                   # Unit tests
├── requirements.txt         # Dependencies
├── README.md               # Main documentation (progressive)
├── CLAUDE.md               # Claude Code instructions
└── todo.md                 # This file
```

---

## Notes
- Each file must have Google-style docstrings
- Follow SOLID and DRY principles
- No file > 1000 lines
- Use relative paths
- Keep it simple - minimal changes per step

---

## Review Section
(To be completed after implementation)
