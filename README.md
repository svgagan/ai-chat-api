# AI Chat API

A production-structured AI API built with FastAPI and LiteLLM.
Switch AI providers by changing two lines in `.env`. Zero application code changes.

---

## What Is Built So Far

| Project | What It Does |
|---|---|
| Project 1 — AI Chat | Model-agnostic chat API with system prompt support |
| Project 2 — Structured Output | Extract validated JSON from unstructured text |
| Project 3 — Streaming | Stream AI responses token by token using SSE |
| Project 4 — Conversation Memory | Stateful multi-turn chat with session management |
| Project 5 — Semantic Search | Embed and search documents by meaning using pgvector |

---

## Tech Stack

```
Python          FastAPI         LiteLLM
Instructor      Groq            Ollama
PostgreSQL      pgvector        SQLAlchemy
```

---

## Prerequisites

- Python 3.11+
- Docker (for PostgreSQL with pgvector)
- Ollama (for local embeddings)
- A free Groq API key from console.groq.com

---

## Setup

### 1 — Clone the repo

```bash
git clone <your-repo-url>
cd ai-chat-api
```

### 2 — Create virtual environment

```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Start PostgreSQL with pgvector

```bash
docker run -d \
  --name pgvector-db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=ai_chat \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

Enable the pgvector extension:

```bash
docker exec -it pgvector-db psql -U postgres -d ai_chat
```

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE document_chunks (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    content         TEXT        NOT NULL,
    embedding       vector(768) NOT NULL,
    source          TEXT        NOT NULL,
    chunk_index     INTEGER     NOT NULL DEFAULT 0,
    content_hash    TEXT,
    is_deleted      BOOLEAN     DEFAULT FALSE,
    embedding_model TEXT        DEFAULT 'ollama/nomic-embed-text',
    metadata        JSONB       DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    created_by      TEXT
);
CREATE INDEX ON document_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON document_chunks (content_hash);
CREATE INDEX ON document_chunks (source);
CREATE INDEX ON document_chunks (is_deleted) WHERE is_deleted = FALSE;
\q
```

### 5 — Start Ollama for local embeddings

```bash
# Install Ollama
brew install ollama

# Pull embedding model
ollama pull nomic-embed-text

# Start as background service
brew services start ollama
```

### 6 — Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your values:

```bash
# Chat model
AI_MODEL=groq/llama-3.1-8b-instant
AI_API_KEY=your_groq_key_here

# Embedding model — local via Ollama, no key needed
EMBEDDING_MODEL=ollama/nomic-embed-text
EMBEDDING_API_KEY=

# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_chat

# Memory settings
MAX_CONVERSATION_MESSAGES=20
```

### 7 — Run

```bash
uvicorn app.main:app --reload
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API documentation.

---

## API Reference

### Chat

```
POST /api/v1/chat
```

General purpose chat. Accepts a message and optional system prompt.

```json
{
  "message": "What is the capital of India?",
  "system_prompt": "You are a helpful assistant."
}
```

---

```
POST /api/v1/explain
```

Explains any topic tailored to a specific audience. System prompt is built internally — the caller never controls AI behavior directly.

```json
{
  "topic": "black holes",
  "audience": "5 year old"
}
```

---

### Structured Output

```
POST /api/v1/extract
```

Extracts structured data from unstructured customer support text. Returns validated JSON with name, age, order number, email, issue, and sentiment. Uses Pydantic schema enforcement via Instructor with automatic retry on validation failure.

```json
{
  "text": "Hi I am John Smith. Order 4521 hasn't arrived. Pretty annoyed. john@email.com"
}
```

---

### Streaming

```
POST /api/v1/stream-chat
```

Streams AI response token by token using Server-Sent Events. Test with curl:

```bash
curl -X POST http://localhost:8000/api/v1/stream-chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me a short story about a robot."}' \
  --no-buffer
```

---

### Conversation Memory

```
POST /api/v1/conversation
```

Stateful multi-turn chat. Pass a `session_id` to continue an existing conversation. Server generates one if not provided. Echo it back from the response and send it with every subsequent message.

```json
{
  "message": "My name is Arjun.",
  "session_id": "optional-provide-or-server-generates"
}
```

```
DELETE /api/v1/conversation/{session_id}
```

Clears all history for a session. Use when the user starts a new conversation or logs out.

---

### Semantic Search

```
POST /api/v1/documents
```

Embeds a text chunk and stores it in pgvector. Skips duplicates automatically using content hash.

```json
{
  "content": "Our return policy allows returns within 30 days of purchase.",
  "source": "return_policy.txt",
  "chunk_index": 0
}
```

---

```
POST /api/v1/search
```

Semantic search over indexed documents. Finds results by meaning, not keywords. Optionally restrict to a specific source document.

```json
{
  "query": "how long do I have to send something back?",
  "limit": 5,
  "source_filter": "return_policy.txt"
}
```

---

```
DELETE /api/v1/documents/{source}
```

Soft deletes all chunks from a source. Data is preserved in the database but excluded from all search results.

---

```
GET /health
```

Health check endpoint. Returns `{"status": "healthy"}`. Used by load balancers and monitoring tools.

---

## Project Structure

```
ai-chat-api/
├── venv/
├── .env                           secrets, never committed
├── .env.example                   committed, shows required keys
├── .gitignore
├── requirements.txt
└── app/
    ├── main.py                    FastAPI app, all routers registered
    ├── config.py                  single source of truth for config
    ├── database.py                SQLAlchemy engine and session setup
    ├── models/
    │   └── document_chunks.py     SQLAlchemy model for vector table
    ├── schemas/
    │   └── customer_inquiry.py    Pydantic schema for extraction
    ├── services/
    │   ├── ai_service.py          all AI logic — chat, extract, stream
    │   ├── conversation_service.py session and memory management
    │   └── embedding_service.py   embed, index, search, soft delete
    └── routers/
        ├── chat.py                /chat, /explain
        ├── extract.py             /extract
        ├── stream.py              /stream-chat
        ├── conversation.py        /conversation
        └── search.py              /documents, /search
```

---

## Switching AI Providers

Change only `.env` — zero application code changes:

```bash
# Groq (current)
AI_MODEL=groq/llama-3.1-8b-instant
AI_API_KEY=your_groq_key

# OpenAI
AI_MODEL=openai/gpt-4o
AI_API_KEY=your_openai_key

# Anthropic
AI_MODEL=anthropic/claude-3-5-sonnet-20241022
AI_API_KEY=your_anthropic_key

# Local via Ollama
AI_MODEL=ollama/llama3.1
AI_API_KEY=
```

Same for embedding model:

```bash
# Local via Ollama (current)
EMBEDDING_MODEL=ollama/nomic-embed-text
EMBEDDING_API_KEY=

# OpenAI
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_API_KEY=your_openai_key
```

---

## Design Decisions

**Service Layer pattern** — Routers handle HTTP only. Services handle AI and business logic only. Neither knows about the other's concerns. Swapping LiteLLM for a different library means changing one file.

**Model-agnostic configuration** — The application never hardcodes a provider name. LiteLLM routes to the correct provider based on the model string prefix. One generic `AI_API_KEY` covers any provider.

**Caller never controls system prompts** — System prompts are business decisions built internally per endpoint. Exposing them to callers is a prompt injection risk — same root cause as SQL injection.

**Temperature not exposed to callers** — Temperature is set per use case in config. Caller-controlled temperature means anyone can break production behavior by setting it to 1.8.

**Schema enforcement via Instructor** — Structured output uses protocol-level schema enforcement, not just a system prompt instruction. Automatic retry with error feedback if validation fails. Cap at 2 retries.

**Optional fields for extraction** — Every extracted field is Optional. Real-world customer messages rarely contain every possible field. Required fields reject valid messages.

**Soft delete** — Documents are never hard deleted. `is_deleted` flag is set. Data is preserved for audit trails and accidental deletion recovery.

**Content hashing for deduplication** — SHA256 hash computed before every embedding call. Identical content from the same source is never re-embedded. Saves API calls and prevents duplicate search results.

**TIMESTAMPTZ not TIMESTAMP** — All timestamps are timezone-aware and stored as UTC. Plain TIMESTAMP strips timezone information and causes subtle bugs in distributed systems.

**Embedding model stored per row** — Enables safe migration when upgrading models. Re-embed old rows in the background while new rows use the new model, both coexisting until migration is complete.

---

## Environment Variables Reference

```bash
# Required
AI_MODEL=groq/llama-3.1-8b-instant
AI_API_KEY=your_api_key

# Required for Project 5
EMBEDDING_MODEL=ollama/nomic-embed-text
EMBEDDING_API_KEY=
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_chat

# Optional with defaults
MAX_CONVERSATION_MESSAGES=20
```