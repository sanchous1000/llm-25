# Lab 2 Report - RAG Agent from Documentation

## 1. Assignment Description

This laboratory work implements a RAG (Retrieval-Augmented Generation) agent system for querying documentation. The work focuses on:

- Parsing and normalizing documents from various formats (PDF, DOCX, PPTX, HTML, MD) into Markdown
- Implementing configurable chunking strategies with metadata preservation
- Building vector embeddings (dense, sparse, hybrid) for semantic search
- Deploying a vector store (Chroma) with HNSW indexing
- Evaluating retrieval quality with standard metrics (Recall@k, Precision@k, MRR)
- Creating a complete RAG pipeline with citation support
- Comparing multiple configurations to optimize performance

The work includes five main components:

1. **Document Parsing** (`scripts/parse_to_markdown.py`) - Converts source documents to normalized Markdown with metadata
2. **Index Building** (`scripts/build_index.py`) - Chunks documents and creates embeddings/vector indices
3. **Vector Store Loading** (`scripts/load_to_vector_store.py`) - Loads and manages the vector database
4. **RAG Pipeline** (`scripts/rag_pipeline.py`) - Complete query-answering system with retrieval and generation
5. **Evaluation** (`scripts/evaluate.py`) - Automated quality assessment with retrieval metrics
6. **Configuration Comparison** (`scripts/compare_configurations.py`) - Tests and compares different parameter settings

## 2. Technologies and Models Used

### Platforms and Services:

- **Chroma** - Vector database for storing embeddings with HNSW indexing
- **LangChain** - Framework for building RAG applications
- **Ollama** - Local LLM server for answer generation

### Embedding Models:

- **sentence-transformers/all-MiniLM-L6-v2** - Dense embedding model (default)
- **BM25** - Sparse retrieval via rank_bm25
- **Hybrid** - Combination of dense and sparse retrieval using Reciprocal Rank Fusion (RRF)

### LLM Models:

- **gemma3:4b** - Language model for answer generation (configurable via config.yaml)

### Document Processing:

- **PyPDFLoader** - PDF document parsing
- **Docx2txtLoader** - DOCX document parsing
- **python-pptx** - PPTX presentation parsing
- **UnstructuredHTMLLoader** - HTML document parsing
- **markdownify** - HTML to Markdown conversion

## 3. Results

### System Architecture

The system implements a complete RAG pipeline with the following stages:

1. **Document Parsing**: Source documents (PDF, DOCX, PPTX, HTML, MD) are converted to normalized Markdown format with YAML frontmatter containing:
   - Source file name and path
   - Document type
   - Page/slide count
   - Modification date
   - Original metadata

2. **Chunking Strategy**: Documents are split using configurable strategies:
   - **Recursive Character Splitter** - Splits by characters with respect to document structure
   - **Token Splitter** - Splits by token count
   - **Markdown Header Splitter** - Splits by markdown headers (H1-H3)
   - **Hybrid Approach** - First splits by headers, then by size/overlap

   All strategies preserve:
   - Header hierarchy (Header 1, Header 2, Header 3)
   - Page numbers (extracted from "## Page X" markers)
   - Slide numbers (extracted from "## Slide X" markers)
   - Source metadata

3. **Vectorization**: Three approaches are supported:
   - **Dense**: Sentence transformer embeddings stored in Chroma
   - **Sparse**: BM25 retriever with pickle serialization
   - **Hybrid**: Combination using RRF (Reciprocal Rank Fusion) with configurable weights

4. **Vector Store**: Chroma database with HNSW indexing:
   - Configurable HNSW parameters (M, ef_construction, ef_search)
   - Support for cosine similarity
   - Persistent storage with collection management

5. **RAG Pipeline**: Complete query-answering system:
   - Query vectorization (dense/sparse/hybrid)
   - Top-k retrieval with configurable k
   - Context assembly with citations
   - LLM prompt construction
   - Answer generation with source attribution

6. **Evaluation**: Automated quality assessment:
   - **Recall@k**: Fraction of ground truth chunks found in top-k results
   - **Precision@k**: Fraction of top-k results that are relevant
   - **MRR**: Mean Reciprocal Rank of first relevant result
   - Comparison across multiple configurations

### Implementation Features

- **Configurable Parameters**: All key parameters are configurable via `config.yaml`:
  - Chunk size (100-1000 tokens)
  - Chunk overlap
  - Splitter type (recursive/token/markdown_only)
  - Header inclusion
  - Vectorization type (dense/sparse/hybrid)
  - Embedding model
  - HNSW parameters

- **Idempotent Operations**: Scripts support rebuild flags (`--rebuild`, `--drop-and-reindex`) for safe re-indexing without manual cleanup

- **Citation Format**: Improved citation system includes:
  - Source file name
  - Page/slide numbers
  - Section hierarchy (Header 1 > Header 2 > Header 3)
  - Content snippets
  - Formatted as: `Source | Page X | Section > Subsection`

- **Ground Truth Evaluation**: 20 representative questions with ground truth chunk identifiers for automated metric calculation

- **Configuration Comparison**: Automated testing of multiple configurations with ranking by performance metrics

## 4. Conclusions

A complete RAG system has been implemented that meets all Lab 2 requirements. The system supports multiple document formats, configurable chunking strategies, and three vectorization approaches (dense, sparse, hybrid). The evaluation framework provides standard retrieval metrics (Recall@k, Precision@k, MRR) for quality assessment.

The modular architecture allows easy experimentation with different parameters. The configuration comparison tool enables systematic optimization of chunk size, overlap, splitter type, and vectorization method. The improved citation system provides transparent source attribution with page/slide numbers and section hierarchy.

All components support idempotent operations with rebuild flags, making it easy to experiment with different configurations without manual cleanup. The system is ready for production use with proper error handling and metadata preservation throughout the pipeline.

## 5. Running Instructions

### Prerequisites

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Set up Ollama** (if not already running):

```bash
# Start Ollama server (default: http://localhost:11434)
ollama serve
```

3. **Configure the system** (optional):

Edit `config.yaml` to adjust:
- Chunk size and overlap
- Splitter type
- Vectorization method
- Embedding model
- HNSW parameters
- LLM model

### Step-by-Step Instructions

#### Step 1: Parse Documents to Markdown

Place your source documents (PDF, DOCX, PPTX, HTML, MD) in the `data/` directory, then run:

```bash
python scripts/parse_to_markdown.py
```

**Result**: Documents are converted to normalized Markdown format in `data/markdown/` with YAML frontmatter containing metadata.

#### Step 2: Build Index

Build the vector index with embeddings:

```bash
python scripts/build_index.py --rebuild
```

**Options**:
- `--rebuild` or `--drop-and-reindex`: Rebuild the index from scratch (removes existing index)

**Result**: Creates vector store in `chroma_db/` with:
- Dense embeddings in Chroma (if using dense/hybrid)
- BM25 retriever pickle file (if using sparse/hybrid)

#### Step 3: Load to Vector Store (Alternative)

Alternatively, use the separate load script:

```bash
python scripts/load_to_vector_store.py --rebuild
```

**Options**:
- `--rebuild` or `--drop-and-reindex`: Rebuild the index
- `--verify`: Verify that the vector store is properly loaded

#### Step 4: Query the RAG System

Ask questions using the RAG pipeline:

```bash
python scripts/rag_pipeline.py --query "What is the scope of the sustainability statement?"
```

**Result**: Returns answer with citations showing:
- Source file
- Page/slide numbers
- Section hierarchy
- Content snippets

#### Step 5: Evaluate Quality

Run evaluation with retrieval metrics:

```bash
python scripts/evaluate.py
```

**Options**:
- `--questions <path>`: Use custom evaluation questions JSON file

**Result**: Calculates and displays:
- Average Recall@5 and Recall@10
- Average Precision@5 and Precision@10
- Average MRR (Mean Reciprocal Rank)
- Detailed results saved to `evaluation_results.json`

#### Step 6: Compare Configurations

Test multiple configurations and compare performance:

```bash
python scripts/compare_configurations.py --default
```

**Options**:
- `--default`: Use default set of configurations
- `--configs <file.json>`: Use custom configurations from JSON file
- `--no-rebuild`: Don't rebuild index for each config (use existing)

**Result**: 
- Tests all configurations
- Ranks by MRR
- Saves detailed comparison to `configuration_comparison.json`

### Example Workflow

```bash
# 1. Parse documents
python scripts/parse_to_markdown.py

# 2. Build index with default config
python scripts/build_index.py --rebuild

# 3. Evaluate current configuration
python scripts/evaluate.py

# 4. Compare different configurations
python scripts/compare_configurations.py --default

# 5. Query the system
python scripts/rag_pipeline.py --query "What ISO certifications does the company maintain?"
```

### Configuration Examples

**Small chunks for detailed retrieval**:
```yaml
data:
  chunk_size: 300
  chunk_overlap: 30
splitter:
  type: "recursive"
  include_headers: true
vectorization:
  type: "dense"
```

**Large chunks for context**:
```yaml
data:
  chunk_size: 800
  chunk_overlap: 80
splitter:
  type: "recursive"
  include_headers: true
vectorization:
  type: "hybrid"
```

**Markdown-only splitting**:
```yaml
splitter:
  type: "markdown_only"
  include_headers: true
vectorization:
  type: "dense"
```

### File Structure

```
lab2/
├── config.yaml                    # Configuration file
├── requirements.txt               # Python dependencies
├── evaluation_questions.json     # 20 evaluation questions with ground truth
├── data/
│   ├── markdown/                 # Parsed Markdown documents
│   └── *.pdf, *.docx, etc.      # Source documents
├── chroma_db/                    # Vector store (created by build_index.py)
├── scripts/
│   ├── parse_to_markdown.py      # Document parsing
│   ├── build_index.py            # Index building
│   ├── load_to_vector_store.py   # Vector store loading
│   ├── rag_pipeline.py           # RAG query system
│   ├── evaluate.py               # Quality evaluation
│   └── compare_configurations.py # Configuration comparison
└── README.md                     # This file
```

### Troubleshooting

**Issue**: Vector store not found
- **Solution**: Run `python scripts/build_index.py --rebuild`

**Issue**: BM25 retriever not found
- **Solution**: Ensure `vectorization.type` is set to "sparse" or "hybrid" in config.yaml, then rebuild

**Issue**: Ollama connection error
- **Solution**: Ensure Ollama server is running on `http://localhost:11434` or update `llm.base_url` in config.yaml

**Issue**: Import errors
- **Solution**: Install all dependencies: `pip install -r requirements.txt`

**Issue**: Permission denied when rebuilding
- **Solution**: Close any processes using the vector store, or wait a moment and retry

