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

### Phase 7: LLM Integration (Free/Local)
- [ ] **Document**: LLM choice (Ollama vs HuggingFace, model selection)
- [ ] Set up Ollama (Llama 3.2 or Mistral) OR HuggingFace model
- [ ] Create prompt template for article Q&A (modify prompt.py)
- [ ] Implement answer generation with retrieved context
- [ ] Handle context window limits
- [ ] **Test**: Ask questions, verify answer quality
- [ ] **Update** project structure in README

### Phase 8: Gradio UI
- [ ] **Document**: UI design (input, outputs, features)
- [ ] Set up Gradio interface (reuse app.py structure)
- [ ] Input: Query textbox
- [ ] Output 1: LLM-generated answer
- [ ] Output 2: Retrieved article cards (title, date, excerpt, link)
- [ ] Output 3: Image gallery for retrieved images
- [ ] Add retrieval mode selector (BM25/Semantic/Combined)
- [ ] **Test**: Full user flow - query → answer + images + articles
- [ ] **Update** project structure in README

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

## Technology Stack (Free/Local)

### Core Components
- **Web Scraping**: BeautifulSoup4 + requests
- **Text Embeddings**: sentence-transformers (all-MiniLM-L6-v2, 384-dim)
- **Image Embeddings**: CLIP (openai/clip-vit-base-patch32, 512-dim)
- **Vector Database**: Qdrant (local, advanced filtering with arrays & date ranges)
- **BM25**: rank_bm25 with NLTK tokenization
- **LLM**: Ollama (Llama 3.2) or HuggingFace Transformers (pending)
- **UI**: Gradio (pending)
- **Chunking**: langchain RecursiveCharacterTextSplitter (500 chars, 50 overlap)

### File Reuse from RAG/
- ✅ `retriever.py` - 70% reusable (BM25 + semantic logic)
- ✅ `tokenizing.py` - 90% reusable
- ✅ `app.py` - 60% reusable (Gradio structure)
- 🔄 `LLM_usage.py` - 40% reusable (replace Groq with Ollama)
- 🔄 `prompt.py` - 20% reusable (change prompt content)

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
