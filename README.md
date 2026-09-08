# rag-pinecone

A production-style Retrieval-Augmented Generation (RAG) API built with **FastAPI**, **Pinecone**, **OpenAI**, and **LangChain**. It ingests documents from URLs or PDFs, indexes them as embeddings in Pinecone, and answers questions grounded in that content — with single-turn Q&A, multi-turn chat with session history, Redis-backed response caching, and built-in RAGAS evaluation.

## Features

- **Ingestion** — scrape and chunk web pages, or upload and chunk PDFs, then upsert as embeddings into Pinecone
- **Query** — ask a question and get an LLM answer grounded in the top-k most relevant chunks, with cited sources
- **Chat** — conversational RAG with per-session history (`session_id`), including endpoints to view or clear history
- **Evaluation** — score answers with [RAGAS](https://github.com/explodinggradients/ragas) (faithfulness, answer relevancy, context precision) for a single QA pair or a full dataset
- **Caching** — Redis-backed cache for repeated queries (1-hour TTL), with graceful fallback if Redis is unavailable
- **Auth & rate limiting** — API key auth (`X-API-Key` header) on all non-health routes, plus request rate limiting via `slowapi`
- **Observability** — structured logging (`structlog`), request timing headers, and Prometheus metrics at `/metrics`
- **Ops-ready** — health check, index stats, and debug endpoints; multi-stage Dockerfile; deploy configs for AWS ECS, Railway, and Heroku-style Procfile

## Architecture

```
Client
  │
  ▼
FastAPI app (app/main.py)
  ├── /health                  → liveness/readiness
  ├── /stats                   → Pinecone index stats
  ├── /api/v1/ingest/url       → scrape → chunk → embed → upsert
  ├── /api/v1/ingest/pdf       → parse → chunk → embed → upsert
  ├── /api/v1/query            → embed question → similarity search → LLM answer
  ├── /api/v1/chat             → same as query, + conversation history
  └── /api/v1/eval             → RAGAS scoring (single pair or dataset)

Pinecone   ← vector store (serverless index, cosine similarity)
OpenAI     ← embeddings (text-embedding-3-small) + chat completion (gpt-4o-mini)
Redis      ← optional response cache
```

## Tech stack

| Layer | Tech |
|---|---|
| API framework | FastAPI, Uvicorn |
| Orchestration | LangChain (`langchain`, `langchain-openai`, `langchain-pinecone`) |
| Vector store | Pinecone (serverless) |
| LLM / embeddings | OpenAI |
| Cache | Redis |
| Evaluation | RAGAS |
| Rate limiting | slowapi |
| Metrics | prometheus-fastapi-instrumentator |
| Logging | structlog |
| Testing | pytest |

## Project structure

```
app/
├── main.py                 # FastAPI app, middleware, routers, lifespan
├── api/
│   ├── health.py            # /health, /stats, /debug/*
│   ├── ingest.py             # /ingest/url, /ingest/pdf
│   ├── query.py               # /query
│   ├── chat.py                 # /chat, /chat/{session_id}/history
│   └── eval.py                  # /eval/single, /eval/dataset
├── core/
│   ├── config.py             # pydantic-settings, env-driven config
│   ├── security.py            # API key verification
│   └── logging.py              # structlog setup
├── models/
│   └── schemas.py           # Pydantic request/response models
└── services/
    ├── pinecone_service.py   # index lifecycle, upsert, similarity search
    ├── rag_service.py         # single-turn RAG pipeline
    ├── chat_service.py         # multi-turn RAG pipeline + session store
    ├── cache_service.py         # Redis caching
    └── eval_service.py           # RAGAS evaluation

ingestion/
└── pipeline.py              # URL/PDF/text loading + chunking

tests/
└── test_api.py               # API tests (mocked Pinecone)
```

## Getting started

### Prerequisites

- Python 3.12
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [Pinecone API key](https://www.pinecone.io/)
- Redis (optional — the app runs fine without it; caching is just disabled)

### Setup

```bash
git clone https://github.com/nithin-mani/rag-pinecone.git
cd rag-pinecone

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
# App
APP_ENV=development
APP_PORT=8000
VALID_API_KEYS=dev-key

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini

# Pinecone
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=rag-index
PINECONE_NAMESPACE=default
PINECONE_EMBEDDING_DIM=1536

# RAG
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K=5
SIMILARITY_THRESHOLD=0.70

# Redis (optional)
REDIS_URL=redis://localhost:6379/0
```

The Pinecone index is created automatically on startup if it doesn't already exist (serverless, cosine metric, dimension matching your embedding model).

### Run locally

```bash
uvicorn app.main:app --reload --port 8000
```

- API docs (Swagger): `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Health check: `http://localhost:8000/health`
- Metrics: `http://localhost:8000/metrics`

All routes except `/health` and `/metrics` require an `X-API-Key` header matching one of the comma-separated values in `VALID_API_KEYS`.

## API usage

### Ingest a URL

```bash
curl -X POST http://localhost:8000/api/v1/ingest/url \
  -H "X-API-Key: dev-key" \
  -H "Content-Type: application/json" \
  -d '{
        "urls": ["https://en.wikipedia.org/wiki/Retrieval-augmented_generation"],
        "source_type": "wikipedia"
      }'
```

### Ingest a PDF

```bash
curl -X POST http://localhost:8000/api/v1/ingest/pdf \
  -H "X-API-Key: dev-key" \
  -F "file=@document.pdf"
```

### Ask a question

```bash
curl -X POST http://localhost:8000/api/v1/query/ \
  -H "X-API-Key: dev-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is retrieval augmented generation?", "top_k": 5}'
```

### Chat (multi-turn)

```bash
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "X-API-Key: dev-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "session_id": "user-123"}'
```

Retrieve or clear a session:

```bash
curl http://localhost:8000/api/v1/chat/user-123/history -H "X-API-Key: dev-key"
curl -X DELETE http://localhost:8000/api/v1/chat/user-123 -H "X-API-Key: dev-key"
```

### Evaluate an answer (RAGAS)

```bash
curl -X POST http://localhost:8000/api/v1/eval/single \
  -H "X-API-Key: dev-key" \
  -H "Content-Type: application/json" \
  -d '{
        "question": "What is RAG?",
        "answer": "RAG combines retrieval with generation.",
        "contexts": ["RAG is a method that retrieves documents..."],
        "ground_truth": "RAG stands for Retrieval Augmented Generation."
      }'
```

`/api/v1/eval/dataset` runs the full RAG pipeline over a list of questions and scores the results automatically.

## Configuration reference

All settings are environment-driven (see `app/core/config.py`), loaded via `pydantic-settings` from `.env` or the environment.

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | Environment name |
| `APP_HOST` / `APP_PORT` | `0.0.0.0` / `8000` | Bind address |
| `VALID_API_KEYS` | `dev-key` | Comma-separated list of accepted API keys |
| `OPENAI_API_KEY` | — | Required |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat/completion model |
| `PINECONE_API_KEY` | — | Required |
| `PINECONE_ENVIRONMENT` | `us-east-1` | Serverless region |
| `PINECONE_INDEX_NAME` | `rag-index` | Index name (auto-created) |
| `PINECONE_NAMESPACE` | `default` | Pinecone namespace |
| `PINECONE_EMBEDDING_DIM` | `1536` | Must match the embedding model's output dimension |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `64` | Text splitting parameters |
| `TOP_K` | `5` | Default number of chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | `0.70` | Minimum similarity score for a chunk to be used |
| `REDIS_URL` | `redis://localhost:6379/0` | Optional response cache |

## Testing

```bash
pytest
```

Tests mock Pinecone and Redis, so no live credentials are required to run the suite.

## Deployment

The repo ships with configs for three deployment targets:

- **Docker** — multi-stage `Dockerfile` (builder + slim runtime, non-root user, built-in `HEALTHCHECK` against `/health`)
  ```bash
  docker build -t rag-pinecone .
  docker run -p 8000:8000 --env-file .env rag-pinecone
  ```
- **AWS ECS** — `.github/workflows/deploy.yml` runs tests, builds/pushes to ECR, and deploys via `ecs-task-definition.json` on every push to `main`
- **Railway** — `railway.json` (Nixpacks build, health check on `/health`)
- **Heroku-style PaaS** — `Procfile` for platforms that support it

## Notes

- The `/debug/*` endpoints (`/debug/search`, `/debug/redis`, `/debug/cache-test`) are for local troubleshooting only — consider removing or gating them behind an environment check before deploying to production.
- Chat session history is stored in-memory (`app/services/chat_service.py`), so it does not persist across restarts or scale across multiple instances. Swap in Redis or a database for production multi-instance deployments.
