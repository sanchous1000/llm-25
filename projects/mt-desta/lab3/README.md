# Lab 3 - Langfuse Integration for LLM Request Logging

## Lab Report

### 1. Task Description

This laboratory work implements a comprehensive monitoring and tracing system for LLM applications using the **Langfuse** platform. The work focuses on:

- Integrating Langfuse for tracking RAG system operations
- Creating datasets for system quality evaluation
- Tracing individual LLM requests (queries to Ollama)
- Detailed tracing of RAG requests with step-by-step breakdown
- Automatic retrieval quality evaluation using Langfuse experiments

The implementation consists of four main components:

1. **Dataset Creation** (`load_dataset_to_langfuse.py`) - Converts ground truth data into Langfuse datasets for experiments

2. **Quality Evaluation** (`evaluate_with_langfuse.py`) - Runs experiments with automatic metric calculation

3. **RAG Pipeline Tracing** (`rag_pipeline_with_langfuse.py`) - Detailed tracing of all RAG pipeline stages

4. **Experiment Management** - Automatic evaluation of retrieval quality on datasets with metric computation (Recall, Precision, MRR)

### 2. Technologies and Models Used

#### Platforms and Services:

- **Langfuse** - Platform for monitoring, tracing, and evaluation of LLM applications (deployed locally via Docker Compose)
- **Ollama** - Local server for running LLM models
- **Chroma** - Vector database for document storage and retrieval
- **BM25** - Sparse retrieval algorithm for keyword-based search

#### LLM and Embedding Models:

- **sentence-transformers/all-MiniLM-L6-v2** - Embedding model for creating dense vector representations
- **gemma3:4b** - Language model for answer generation (via Ollama)
- **BM25 Retriever** - Sparse retrieval for keyword-based document search

#### Configuration:

- **Vectorization Type**: Sparse (BM25) retrieval
- **Chunk Size**: 500 characters
- **Chunk Overlap**: 50 characters
- **Retrieval**: Top-k retrieval (k=10 by default)

### 3. Results

#### Monitoring System Architecture

The system implements a complete monitoring cycle for LLM applications:

1. **Dataset Creation**: Ground truth data collected in Lab 2 is converted into a structured Langfuse dataset, where each item contains a question (input) and a list of relevant document identifiers (expected_output).

2. **Request Tracing**: All LLM requests are tracked through Langfuse with preservation of:
   - Input data (prompts, parameters)
   - Output data (model responses)
   - Metadata (model, user, session, timestamps)
   - Performance metrics (duration, token counts)

3. **Detailed RAG Tracing**: The RAG pipeline is broken down into separate observable stages:
   - **document_retrieval** - Document retrieval from vector store
   - **llm_generation** - Answer generation by the language model
   - Each stage is tracked with its own metadata, timestamps, and performance metrics

4. **Experiments**: Automatic evaluation of retrieval quality on datasets with metric computation (Recall@k, Precision@k, MRR).

#### Implementation Features

Each stage of the RAG pipeline is tracked as a separate observation with type:
- `generation` - For generation operations (LLM responses)
- `span` - For intermediate operations (document retrieval)

All observations are linked through `trace_id`, allowing visualization of the complete request execution path.

Each request is enriched with metadata:
- `user_id` - User identifier
- `session_id` - Session identifier (auto-generated UUID)
- `query_start_time`, `query_end_time` - ISO 8601 timestamps
- `retrieval_duration`, `llm_duration` - Performance metrics
- `environment` - Environment identifier (development/production)
- `retrieval_type` - Type of retrieval (sparse/dense/hybrid)
- `model` - LLM model name
- `num_retrieved_docs` - Number of retrieved documents

#### Evaluation Results

The system was evaluated on 20 questions with the following average metrics:

- **Recall@5**: 0.567 (56.7%) - Finds most relevant documents in top 5
- **Precision@5**: 0.330 (33.0%) - About 1 in 3 of top 5 are relevant
- **Recall@10**: 0.667 (66.7%) - Finds most relevant documents in top 10
- **Precision@10**: 0.245 (24.5%) - About 1 in 4 of top 10 are relevant
- **MRR**: 0.723 (72.3%) - Relevant documents appear early in ranking

These results demonstrate:
- **Strong Recall**: 66.7% at k=10 means most relevant documents are retrieved
- **Moderate Precision**: 33% at k=5 suggests some irrelevant documents are included
- **Good Ranking**: MRR of 0.723 indicates relevant documents tend to rank highly

### 4. Conclusions

A complete integration of the monitoring platform with the existing RAG system has been implemented, enabling tracking of all aspects of the application. The RAG pipeline is broken down into logical stages, each tracked separately. This allows identification of performance bottlenecks, analysis of each component's quality, and debugging of issues at specific stages. All requests and responses are saved in a centralized repository with search, filtering, and analysis capabilities.

The implementation provides:
- **Full Observability**: Complete visibility into RAG pipeline execution
- **Performance Monitoring**: Detailed timing breakdowns for each step
- **Quality Evaluation**: Automated metrics calculation for retrieval quality
- **User Analytics**: Session and user-level tracking for analytics
- **Experiment Management**: Easy comparison of different configurations

### 5. Running Instructions

#### Prerequisites

1. **Install Langfuse** (locally via Docker Compose):

```bash
cd lab3/langfuse
docker-compose up -d
```

2. **Create `.env` file** in the `lab3` directory with Langfuse access keys:

```bash
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=http://localhost:3001
USER_ID=default-user
SESSION_ID=optional-session-id
ENVIRONMENT=development
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
pip install -r ../lab2/requirements.txt
```

#### Step-by-Step Instructions

##### Step 1: Deploy Langfuse

```bash
cd lab3/langfuse
docker-compose up -d
```

Access Langfuse UI at: http://localhost:3001

##### Step 2: Create Project and Get API Keys

1. Open http://localhost:3001 in your browser
2. Create an account (first-time setup) or sign in
3. Create a new project
4. Go to Project Settings → API Keys
5. Copy the **Public Key** and **Secret Key** to your `.env` file

##### Step 3: Load Dataset to Langfuse

```bash
python load_dataset_to_langfuse.py --dataset-name "rag-evaluation-dataset"
```

##### Step 4: Trace RAG Request

```bash
python rag_pipeline_with_langfuse.py \
  --query "What is the scope of the sustainability statement?" \
  --user-id "user-123" \
  --session-id "session-abc"
```

**Result**: A trace is created in Langfuse with two observations:

1. `document_retrieval` - Document retrieval from vector store
2. `llm_generation` - Answer generation by LLM

##### Step 5: Evaluate Quality via Experiments

```bash
python evaluate_with_langfuse.py \
  --dataset-name "rag-evaluation-dataset" \
  --experiment-name "baseline-experiment"
```

This will:
1. Load the dataset from Langfuse
2. Run evaluation on each question
3. Calculate metrics: Recall@k, Precision@k, MRR
4. Log all results to Langfuse Experiment Run
5. Display summary statistics

##### Step 6: View Results in Langfuse

1. **Traces**: View individual query traces at http://localhost:3001/traces
   - See full interaction flow: retrieval → LLM call
   - View metadata, timings, and scores
   - Filter by user_id, session_id, or experiment_name

2. **Experiments**: View experiment runs at http://localhost:3001/experiments
   - Compare different configurations
   - Analyze metrics across runs
   - Export results

3. **Datasets**: View loaded datasets at http://localhost:3001/datasets
   - Manage dataset items
   - Run experiments on datasets

---

This lab integrates Langfuse for centralized logging of LLM requests and RAG pipeline interactions.

## Overview

This implementation:
1. Deploys Langfuse locally using Docker Compose
2. Integrates Langfuse logging into the existing lab2 RAG application
3. Logs all user interactions, including intermediate steps (retrieval, LLM calls)
4. Loads evaluation datasets into Langfuse
5. Runs experiments with metrics logging (Recall@k, Precision@k, MRR)

## Prerequisites

- Docker and Docker Compose installed
- Python 3.8+
- Existing lab2 RAG application (vector store and indices must be built)
- LLM server running (e.g., Ollama on localhost:11434)

## Setup

### 1. Deploy Langfuse

Start Langfuse using Docker Compose:

```bash
cd lab3
docker-compose up -d
```

Wait for Langfuse to be ready (check logs with `docker-compose logs -f langfuse`).

Access Langfuse UI at: http://localhost:3001

### 2. Configure Langfuse Project

1. Open http://localhost:3001 in your browser
2. Create an account (first-time setup) or sign in
3. Create a new project
4. Go to Project Settings → API Keys
5. Copy the **Public Key** and **Secret Key**

### 3. Set Environment Variables

Copy the example environment file and fill in your credentials:

```bash
cp env.example .env
```

Edit `.env` and add your Langfuse credentials:

```env
LANGFUSE_PUBLIC_KEY=your-public-key-here
LANGFUSE_SECRET_KEY=your-secret-key-here
LANGFUSE_HOST=http://localhost:3001
USER_ID=default-user
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Running RAG Queries with Langfuse Logging

Use the Langfuse-integrated RAG pipeline to query and automatically log all interactions:

```bash
python rag_pipeline_with_langfuse.py --query "What is the scope of the sustainability statement?"
```

Optional: specify a user ID for session tracking:

```bash
python rag_pipeline_with_langfuse.py --query "Your question" --user-id "user-123"
```

All interactions are automatically logged to Langfuse, including:
- Document retrieval steps
- Retrieved documents and their metadata
- LLM prompts and responses
- Response times and token counts
- Full trace linking all steps

### Loading Dataset to Langfuse

Load the evaluation questions from lab2 into Langfuse as a dataset:

```bash
python load_dataset_to_langfuse.py --dataset-name "rag-evaluation-dataset"
```

Or specify a custom questions file:

```bash
python load_dataset_to_langfuse.py --questions ../lab2/evaluation_questions.json --dataset-name "my-dataset"
```

### Running Evaluation Experiments

Run evaluation experiments with automatic metrics logging to Langfuse:

```bash
python evaluate_with_langfuse.py --dataset-name "rag-evaluation-dataset" --experiment-name "baseline-experiment"
```

This will:
1. Load the dataset from Langfuse
2. Run evaluation on each question
3. Calculate metrics: Recall@k, Precision@k, MRR
4. Log all results to Langfuse Experiment Run
5. Display summary statistics

### Viewing Results in Langfuse

1. **Traces**: View individual query traces at http://localhost:3001/traces
   - See full interaction flow: retrieval → prompt construction → LLM call
   - View metadata, timings, and scores

2. **Experiments**: View experiment runs at http://localhost:3001/experiments
   - Compare different configurations
   - Analyze metrics across runs
   - Export results

3. **Datasets**: View loaded datasets at http://localhost:3001/datasets
   - Manage dataset items
   - Run experiments on datasets

## Architecture

### Integration Approach

The implementation wraps the existing `lab2/scripts/rag_pipeline.py` RAGAgent without modifying it:

- `rag_pipeline_with_langfuse.py`: Wrapper that adds Langfuse logging to lab2's RAGAgent
- `load_dataset_to_langfuse.py`: Loads evaluation questions into Langfuse datasets
- `evaluate_with_langfuse.py`: Runs evaluations with Langfuse Experiment Run integration

### Logging Structure

Each RAG query creates a trace with the following structure:

```
Trace: rag_query
├── Generation: document_retrieval
│   └── Metadata: retrieval type, k, document sources, scores
└── Generation: llm_generation
    └── Metadata: model, prompt length, response length, duration
```

### Metrics Logged

- **Retrieval Metrics**:
  - Recall@k: Fraction of ground truth chunks found in top-k
  - Precision@k: Fraction of top-k chunks that are relevant
  - MRR: Mean Reciprocal Rank

- **LLM Metrics**:
  - Response time
  - Prompt length
  - Response length
  - Model parameters

## Troubleshooting

### Langfuse not accessible

Check if containers are running:
```bash
docker-compose ps
```

Check logs:
```bash
docker-compose logs langfuse
```

### Authentication errors

Verify your `.env` file has correct credentials:
```bash
cat .env
```

Ensure Langfuse project is created and API keys are valid.

### Import errors

Make sure lab2 dependencies are installed:
```bash
pip install -r ../lab2/requirements.txt
pip install -r requirements.txt
```

### Vector store not found

Ensure lab2 vector store is built:
```bash
cd ../lab2
python scripts/build_index.py
```

## Files Structure

```
lab3/
├── docker-compose.yml          # Langfuse deployment
├── env.example                 # Environment variables template
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── rag_pipeline_with_langfuse.py  # RAG wrapper with Langfuse
├── load_dataset_to_langfuse.py    # Dataset loader
└── evaluate_with_langfuse.py      # Experiment evaluator
```

## Next Steps

1. **Compare Configurations**: Run experiments with different retrieval configurations (dense, sparse, hybrid) and compare results in Langfuse
2. **Add Custom Metrics**: Extend the evaluator to include additional metrics (e.g., answer quality, faithfulness)
3. **RAGAS Integration**: Optionally integrate RAGAS framework for advanced RAG metrics
4. **Production Deployment**: Consider deploying Langfuse to a production environment for ongoing monitoring

## References

- [Langfuse Documentation](https://langfuse.com/docs)
- [Langfuse Experiments](https://langfuse.com/docs/evaluation/experiments/experiments-via-sdk)
- [RAGAS Framework](https://docs.ragas.io/)

