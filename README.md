# 📚 Multi-Document RAG System

A production-ready **Retrieval-Augmented Generation** system for academic research papers with cross-document reasoning and RAGAS evaluation.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Chainlit UI │────▶│  FastAPI      │────▶│  Groq LLM API    │
│  (Port 8001) │     │  (Port 8000)  │     │  llama-3.3-70b   │
└──────────────┘     └──────┬───────┘     │  deepseek-r1     │
                            │              └──────────────────┘
                            │
                     ┌──────▼───────┐     ┌──────────────────┐
                     │  Core Engine  │────▶│  Google Gemini    │
                     │  - Ingestion  │     │  Embedding API    │
                     │  - Retrieval  │     │  (3072-dim)       │
                     │  - Generation │     └──────────────────┘
                     │  - Evaluation │
                     └──────┬───────┘
                            │
                     ┌──────▼───────┐
                     │  Qdrant      │
                     │  Vector DB   │
                     │  (Port 6333) │
                     └──────────────┘
```

## Features

- **Multi-format ingestion**: PDF, DOCX, TXT with automatic metadata extraction
- **Smart query routing**: Routes simple vs. complex queries to appropriate models
- **Self-correction**: Detects low-relevance retrievals and rewrites queries automatically
- **Cross-document reasoning**: Compare and contrast findings across multiple papers
- **RAGAS evaluation**: 5 metrics — faithfulness, answer relevancy, context precision, context recall, answer correctness
- **Structured API**: Full REST API with async endpoints throughout

## Prerequisites

- **Python** 3.11+
- **Docker** (for Qdrant vector database)
- **API Keys**:
  - [Groq API Key](https://console.groq.com/)
  - [Google AI API Key](https://aistudio.google.com/apikey)

## Quick Start

### 1. Clone and configure

```bash
git clone <repo-url>
cd multi_doc_rag
cp .env.example .env
# Edit .env and fill in your API keys
```

### 2. Start Qdrant (Docker)

```bash
docker-compose up -d
```

This starts Qdrant on port 6333 (~150MB image).

### 3. Install dependencies

```bash
# Using UV (recommended)
uv venv
uv pip install -r requirements.txt

# Or using pip
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 4. Start FastAPI backend

```bash
uvicorn app.main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### 5. Start Chainlit UI (optional)

```bash
chainlit run chainlit_app/app.py --port 8001
```

## API Usage

### Ingest a document

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@paper.pdf" \
  -F "source=arxiv"
```

### Batch ingest

```bash
curl -X POST http://localhost:8000/api/v1/ingest/batch \
  -F "files=@paper1.pdf" \
  -F "files=@paper2.pdf" \
  -F "source=upload"
```

### Query documents

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What dataset was used in the study?",
    "mode": "standard",
    "top_k": 5
  }'
```

### Compare documents

```bash
curl -X POST http://localhost:8000/api/v1/query/compare \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do the methodologies differ?",
    "doc_ids": ["doc-id-1", "doc-id-2"],
    "aspect": "methodology"
  }'
```

### List documents

```bash
curl http://localhost:8000/api/v1/documents
```

### Delete a document

```bash
curl -X DELETE http://localhost:8000/api/v1/documents/{doc_id}
```

### Run RAGAS evaluation

```bash
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{"sample_size": 10}'
```

### Check evaluation report

```bash
curl http://localhost:8000/api/v1/evaluate/{eval_id}
```

### Health check

```bash
curl http://localhost:8000/api/v1/health
```

## Running Tests

```bash
pytest tests/ -v
```

Tests use mocked external services (Groq, Gemini, Qdrant) so no API keys are needed.

## RAGAS Evaluation

The system includes 20 sample questions across 5 categories:

| Category | Count | Example |
|----------|-------|---------|
| Factual | 4 | "What dataset was used?" |
| Comparative | 4 | "How do methods differ?" |
| Summarization | 4 | "Summarize key findings" |
| Multi-hop | 4 | "What evidence supports the claim?" |
| Out-of-scope | 4 | "What is NVIDIA's stock price?" |

### Running evaluation

1. Ingest at least one document first
2. Use the `/api/v1/evaluate` endpoint or the `/eval` command in Chainlit
3. Reports are saved to `evaluation/reports/`

## Architecture Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| LLM | Groq API | Fast inference, no GPU needed locally |
| Embeddings | Gemini API | High-quality 3072-dim vectors, no local models |
| Vector DB | Qdrant | Lightweight (~150MB), rich filtering, easy setup |
| Framework | FastAPI | Async-native, auto-generated docs, Pydantic validation |
| UI | Chainlit | Minimal setup for prototyping, chat-native interface |
| Parsing | pypdf + python-docx | Lightweight, no heavy dependencies |
| Orchestration | LangChain | Prompt templates and chain composition |

## Troubleshooting

### Qdrant connection refused

```bash
# Ensure Qdrant is running
docker-compose up -d
docker ps  # Check container status
```

### Groq rate limits

The system uses exponential backoff (3 retries). If you hit persistent rate limits:
- Reduce `TOP_K_RETRIEVAL` in `.env`
- Increase delay between requests
- Check your Groq API usage dashboard

### Empty search results

- Ensure documents are ingested (`GET /api/v1/documents`)
- Check Qdrant collection exists
- Verify embedding dimension matches (should be 3072)

### Docker image too large

The Docker image should stay under 500MB. If it grows:
- Ensure no `torch`, `transformers`, or `sentence-transformers` in dependencies
- Check `requirements.txt` for unexpected heavy packages

## Project Structure

```
multi_doc_rag/
├── app/                    # FastAPI application
│   ├── main.py             # Entry point
│   ├── config.py           # Settings
│   ├── api/routes/         # API endpoints
│   ├── core/
│   │   ├── ingestion/      # Document loading, chunking, tagging
│   │   ├── retrieval/      # Embedding, search, routing, self-correction
│   │   ├── generation/     # LLM client, prompts, RAG chains
│   │   └── evaluation/     # RAGAS evaluation pipeline
│   ├── models/             # Pydantic schemas & enums
│   └── utils/              # Logging & helpers
├── chainlit_app/           # Chainlit UI
├── evaluation/             # Test questions & reports
├── tests/                  # Test suite
├── docker-compose.yml      # Qdrant only
├── Dockerfile              # App container (slim)
└── requirements.txt        # Dependencies (no torch!)
```

## License

MIT
