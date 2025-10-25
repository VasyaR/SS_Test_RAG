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
- [ ] Create initial README.md with project overview
- [ ] Set up project structure (folders)
- [ ] Create requirements.txt with dependencies
- [ ] Document technology stack choices in README

### Phase 1: Data Ingestion (The Batch Scraper)
- [ ] **Document**: Web scraping approach in README (BeautifulSoup, structure, pagination)
- [ ] Implement web scraper for The Batch articles
- [ ] Extract: title, date, text content, images URLs, article URL
- [ ] Download and save images locally
- [ ] Save raw articles as JSON
- [ ] **Test**: Scrape 10-20 articles, verify data quality
- [ ] **Update** project structure in README

### Phase 2: Text Processing & Chunking
- [ ] **Document**: Chunking strategy in README (paragraph-based, 200-500 tokens, why)
- [ ] Implement text chunking logic (reuse tokenizing.py pattern)
- [ ] Link chunks to their parent articles
- [ ] Link chunks to associated images (based on proximity in HTML)
- [ ] **Test**: Chunk sample articles, verify quality
- [ ] **Update** project structure in README

### Phase 3: Text Embeddings & BM25
- [ ] **Document**: Text retrieval approach (BM25 + sentence-transformers, hybrid search)
- [ ] Set up sentence-transformers (all-MiniLM-L6-v2 or all-distilroberta-v1)
- [ ] Generate embeddings for text chunks (reuse retriever.py pattern)
- [ ] Set up BM25 index (reuse from RAG/)
- [ ] **Test**: Query "AI trends" and verify text retrieval works
- [ ] **Update** project structure in README

### Phase 4: Image Embeddings (CLIP)
- [ ] **Document**: Image embedding approach (CLIP model, why multimodal)
- [ ] Set up CLIP model (transformers library)
- [ ] Generate embeddings for all images
- [ ] Save embeddings efficiently (split if needed, like RAG/)
- [ ] **Test**: Query with text, retrieve similar images
- [ ] **Update** project structure in README

### Phase 5: Multimodal Vector Database
- [ ] **Document**: Database choice (ChromaDB vs FAISS, metadata filtering)
- [ ] Set up ChromaDB/FAISS for text chunks
- [ ] Store metadata (date, title, article_id, image_path, tags)
- [ ] Add image embeddings to separate collection or same
- [ ] Implement metadata filtering (date, article count)
- [ ] **Test**: Query with filters ("2 latest articles about AI")
- [ ] **Update** project structure in README

### Phase 6: Multimodal Retrieval System
- [ ] **Document**: Fusion strategy (text + image scores, weighting, ranking)
- [ ] Implement multimodal retriever (modify retriever.py)
- [ ] Combine text scores + image scores (weighted fusion)
- [ ] Add metadata filtering integration
- [ ] Return: top K chunks + images + article metadata
- [ ] **Test**: Various queries (text-only, multimodal, with filters)
- [ ] **Update** project structure in README

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
- **Text Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Image Embeddings**: CLIP (openai/clip-vit-base-patch32)
- **Vector Database**: ChromaDB (local, with metadata filtering)
- **BM25**: rank_bm25
- **LLM**: Ollama (Llama 3.2) or HuggingFace Transformers
- **UI**: Gradio
- **Chunking**: langchain TextSplitter or custom

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
