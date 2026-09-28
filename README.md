# AI Chat API

A production-structured AI API built with FastAPI and LiteLLM.
Switch AI providers by changing two lines in `.env`. Zero application code changes.

---

## What Is Built So Far

| Project | What It Does |
|---|---|
| Project 1 - AI Chat | Model-agnostic chat API with system prompt support |
| Project 2 - Structured Output | Extract validated JSON from unstructured text |
| Project 3 - Streaming | Stream AI responses token by token using SSE |
| Project 4 - Conversation Memory | Stateful multi-turn chat with session management |
| Project 5 - Semantic Search | Embed and search documents by meaning using pgvector |
| Project 6 - RAG | Answer questions grounded in retrieved documents, with citations |
| Project 7 - Hybrid Search | Combine semantic and keyword search using Reciprocal Rank Fusion |
| Project 7.5 - Re-ranking | Cross-encoder re-ranking on top of hybrid search results |
| Project 8 - Tool Calling | Let the model call real functions, with a safe execution boundary |
| Project 9 - Multi-Tool Assistant | Correct tool selection among several unrelated tools |
| Project 10 - Agent + MCP | A ReAct reasoning loop, with tools served over five separate MCP servers, built both by hand and with LangChain's create_agent for direct comparison |

---

## Tech Stack

```
Python             FastAPI            LiteLLM
Instructor         Groq               Ollama
PostgreSQL         pgvector           SQLAlchemy
sentence-transformers                 MCP (Model Context Protocol)
LangChain          LangGraph (via langchain's create_agent)
```

---

## Prerequisites

- Python 3.11+
- Docker (for PostgreSQL with pgvector)
- Ollama (for local embeddings)
- A free Groq API key from console.groq.com

---

## Setup

### 1 - Clone the repo

```bash
git clone <your-repo-url>
cd ai-chat-api
```

### 2 - Create virtual environment

```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3 - Install dependencies

```bash
pip install -r requirements.txt
```

### 4 - Start PostgreSQL with pgvector

```bash
docker run -d \
  --name pgvector-db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=ai_chat \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

Enable the pgvector extension and set up full-text search alongside it:

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

-- Vector similarity index, used by semantic search
CREATE INDEX ON document_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON document_chunks (content_hash);
CREATE INDEX ON document_chunks (source);
CREATE INDEX ON document_chunks (is_deleted) WHERE is_deleted = FALSE;

-- Full-text search column, used by keyword search (Project 7)
ALTER TABLE document_chunks
ADD COLUMN content_tsv tsvector
GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX ON document_chunks USING GIN (content_tsv);

\q
```

### 5 - Start Ollama for local embeddings

```bash
brew install ollama
ollama pull nomic-embed-text
brew services start ollama
```

### 6 - Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your values, see the full reference near the bottom of this file.

### 7 - Run the main API

```bash
uvicorn app.main:app --reload
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API documentation.

### 8 - Start the MCP tool servers (required for Project 10's agent)

Project 10's agent discovers its tools from five independently running MCP servers, each simulating a separate team's service. Start each one in its own terminal:

```bash
python mcp_servers/weather_server.py     # port 8001
python mcp_servers/calculator_server.py  # port 8002
python mcp_servers/time_server.py        # port 8003
python mcp_servers/currency_server.py    # port 8004
python mcp_servers/activity_server.py    # port 8005
```

Each one sits idle on its port until a client connects, that is expected. The `/api/v1/agent` and `/api/v1/tool-chat` endpoints will not work correctly until all five are running, since tool discovery happens live against these URLs.

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

Explains any topic tailored to a specific audience. System prompt is built internally, the caller never controls AI behavior directly.

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

Streams AI response token by token using Server-Sent Events.

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

Stateful multi-turn chat. Pass a `session_id` to continue an existing conversation. Server generates one if not provided.

```json
{
  "message": "My name is Arjun.",
  "session_id": "optional-provide-or-server-generates"
}
```

```
DELETE /api/v1/conversation/{session_id}
```

Clears all history for a session.

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

```
POST /api/v1/search
```

Semantic search over indexed documents.

```json
{
  "query": "how long do I have to send something back?",
  "limit": 5,
  "source_filter": "return_policy.txt"
}
```

```
DELETE /api/v1/documents/{source}
```

Soft deletes all chunks from a source.

---

### RAG (Retrieval Augmented Generation)

```
POST /api/v1/ask
```

Answers a question grounded in indexed documents. Uses hybrid search (semantic plus keyword, combined with Reciprocal Rank Fusion) and cross-encoder re-ranking to select context, then generates an answer using only that context. Returns "I don't have information about that" when nothing relevant is found.

```json
{
  "query": "how long do I have to return an item?",
  "source_filter": "return_policy.txt",
  "isChunkCitationRequired": true
}
```

Response includes `answer`, `sources_used`, `chunks_used`, and, when `isChunkCitationRequired` is true, `chunk_citations` showing each chunk's similarity score, RRF score, and rerank score, plus the `rrf_threshold` actually applied. Useful for debugging why a given chunk was or was not used.

```
POST /api/v1/ask-conversational
```

Same as `/ask`, but remembers the conversation and rewrites follow-up questions into standalone queries before retrieval, so "what about the shipping one instead" correctly becomes "what is your shipping policy" before searching.

```json
{
  "session_id": "optional-provide-or-server-generates",
  "query": "What about the shipping one instead?",
  "isChunkCitationRequired": true
}
```

Response includes `rewritten_query`, showing exactly what was searched for, which is essential for debugging when a follow-up question returns unexpected results.

---

### Tool Calling and Agents

```
POST /api/v1/tool-chat
```

A single-turn assistant with access to several unrelated tools (weather, calculator, time, currency conversion, knowledge base search). The model picks the right tool for the question, with no further reasoning loop.

```json
{
  "message": "What is our return policy?"
}
```

```
POST /api/v1/agent
```

A goal-directed ReAct agent. Unlike `/tool-chat`, this handles requests where a later action genuinely depends on an earlier tool's result, for example checking the weather before deciding which activity to suggest. Returns a full step-by-step reasoning trace, not just the final answer, so every tool call and the reasoning behind it is inspectable.

```json
{
  "goal": "Check the weather in Bangalore. Based on the weather, suggest an activity I can do with a budget of 500 INR."
}
```

Response includes `answer`, `steps_taken`, and `trace`, an array where each step shows the model's reasoning text, which tool it called and with what arguments, and the actual observation returned, in order.

```
POST /api/v1/agent-langchain
```

Same goal, same underlying MCP tools, built instead with LangChain's `create_agent`, kept side by side with `/agent` deliberately for comparison rather than as a replacement. Response includes `answer`, `message_count`, and `raw_messages`, each showing role, content, and reasoning (LangChain surfaces this in a different field than the hand-built agent, `additional_kwargs["reasoning_content"]` rather than a top-level `reasoning` attribute).

```json
{
  "goal": "Check the weather in Bangalore. Based on the weather, suggest an activity I can do with a budget of 500 INR."
}
```

---

```
GET /health
```

Health check endpoint. Returns `{"status": "healthy"}`.

---

## Project Structure

```
ai-chat-api/
├── venv/
├── .env                            secrets, never committed
├── .env.example                    committed, shows required keys
├── .gitignore
├── requirements.txt
├── mcp_servers/                     five independent MCP servers,
│   │                                each simulating a separate team's
│   │                                service, zero shared code with app/
│   ├── weather_server.py           port 8001
│   ├── calculator_server.py        port 8002, AST-based safe evaluator
│   ├── time_server.py              port 8003
│   ├── currency_server.py          port 8004
│   └── activity_server.py          port 8005
└── app/
    ├── main.py                     FastAPI app, all routers registered
    ├── config.py                   single source of truth for config
    ├── database.py                 SQLAlchemy engine and session setup
    ├── models/
    │   └── document_chunks.py      SQLAlchemy model for vector table
    ├── schemas/
    │   └── customer_inquiry.py     Pydantic schema for extraction
    ├── tools/
    │   └── knowledge_base_tool.py  the ONLY tool implementation living
    │                               in app/, genuinely coupled to this
    │                               app's database and RAG pipeline,
    │                               so it stays local rather than
    │                               pretending to be a separate service
    ├── services/
    │   ├── ai_service.py           all direct model calls, chat, extract,
    │   │                            stream, tool calling primitives
    │   ├── conversation_service.py session and memory management
    │   ├── embedding_service.py    embed, semantic search, keyword
    │   │                            search, hybrid fusion, re-ranking
    │   ├── rag_service.py          retrieval, threshold filtering,
    │   │                            grounded generation, citations
    │   ├── mcp_client_service.py   discovers tools from the five MCP
    │   │                            servers plus local tools, builds
    │   │                            one merged routing table, routes
    │   │                            every tool call to the right place
    │   ├── tool_service.py         single-turn tool-calling loop,
    │   │                            fully delegates tool access to
    │   │                            mcp_client_service
    │   ├── agent_service.py        ReAct reasoning loop with a full
    │   │                            inspectable trace, hand-built,
    │   │                            delegates tool access to
    │   │                            mcp_client_service
    │   └── langchain_agent_service.py  same MCP tools, built with
    │                                    LangChain's create_agent
    │                                    instead, kept alongside the
    │                                    hand-built version for
    │                                    direct comparison, not as
    │                                    a replacement
    └── routers/
        ├── chat.py                 /chat, /explain
        ├── extract.py               /extract
        ├── stream.py                /stream-chat
        ├── conversation.py          /conversation
        ├── search.py                /documents, /search
        ├── rag.py                   /ask, /ask-conversational
        ├── tools.py                 /tool-chat
        └── agent.py                 /agent, /agent-langchain
```

---

## Switching AI Providers

Change only `.env`, zero application code changes:

```bash
# Groq (current)
AI_MODEL=groq/openai/gpt-oss-20b
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

Same pattern for embeddings and the re-ranking model, each is one config value away from being swapped.

---

## Design Decisions

### From Projects 1 through 5

**Service Layer pattern** - Routers handle HTTP only. Services handle AI and business logic only.

**Model-agnostic configuration** - The application never hardcodes a provider name. One generic `AI_API_KEY` covers any provider, verified directly when Groq deprecated a model mid-project and the fix was a one-line `.env` change, not a code change.

**Caller never controls system prompts** - Prompt injection risk, same root cause as SQL injection.

**Schema enforcement via Instructor** - Protocol-level enforcement, not just a prompt instruction. Automatic retry with error feedback, capped at 2 retries.

**Soft delete everywhere** - Documents are never hard deleted, `is_deleted` is set instead, preserving audit trails.

**Content hashing for deduplication** - SHA256 computed before every embedding call, identical content is never re-embedded twice.

### From Projects 6, 7, and 7.5

**RAG threshold filtering, not blind trust in retrieval** - pgvector always returns the closest matches even when nothing is actually relevant. Chunks are filtered by score before being handed to the model as context, and the endpoint returns an honest "I don't have information about that" rather than forcing an answer from weak context.

**Hybrid search over semantic search alone** - Semantic search is weak at exact identifiers and sometimes ranks a merely-related chunk above the actually-correct one. Keyword search (PostgreSQL native full-text search, no external engine) is combined with semantic search using Reciprocal Rank Fusion, which rewards chunks that both methods agree on.

**RRF query construction uses OR logic, not AND** - PostgreSQL's `plainto_tsquery` defaults to requiring every query word to be present, which fails on natural language questions where an incidental word like "long" never appears in the correct document. Query terms are OR-joined instead, letting `ts_rank` naturally score multi-term matches higher without requiring all terms present.

**Cross-encoder re-ranking with sigmoid activation** - Raw cross-encoder logits are unbounded and not comparable across different queries, confirmed directly by testing, the same score range that indicated a strong match on one query indicated nothing meaningful on another. Sigmoid activation normalizes scores to a consistent 0 to 1 range, though this fixes the scale problem, not a deeper calibration issue where the model's confidence still varies by content type. Re-ranking scores are treated as reliable for reordering candidates within one query, not as an absolute quality bar across different queries.

**Every threshold number is treated as a starting point, not a proven constant** - The similarity threshold, the RRF threshold, `rrf_k`, and the candidate pool multiplier were all either derived from real test data or explicitly flagged as adopted conventions rather than independently benchmarked for this dataset. Anyone reusing this code against different documents should expect to re-verify these numbers, not copy them as fact.

### From Projects 8, 9, and 10

**The model never executes anything, the registry or router is always the security boundary** - Verified directly, not assumed. An early calculator tool used `eval()`, and testing its safety by asking the model to attempt an attack proved nothing, since a refusal and a successful block look identical from the outside. The actual proof came from calling the function directly, bypassing the model entirely. The fix replaced `eval()` with an AST-based allowlist evaluator, which was then verified the same way, direct calls, not model behavior.

**Mocked tools should fail the way the real thing would** - An early mocked weather tool accepted any string as a city name and returned fake data for it, which meant the tool's error-handling path was never actually exercised by testing. A useful mock simulates failure conditions too, not just the happy path.

**Reasoning text location is not guaranteed by the API contract, and varies by provider** - A tool-calling model may put its step-by-step reasoning in `message.content`, in a separate `message.reasoning` field, or omit it depending on the provider and even the specific request. Code that reads reasoning for debugging or trace-building checks multiple possible fields defensively rather than assuming one location.

**One tool-access layer, not two** - Tools are served over the Model Context Protocol, five independent servers each exposing their own capability, discovered live via `list_tools()` rather than hand-written definitions duplicated in application code. A tool genuinely coupled to this app's own database and RAG pipeline (`knowledge_base`) stays local rather than pretending to be an independent service, the honest test for whether something belongs behind MCP is dependency coupling, not naming or domain.

**Synchronous model calls are offloaded to a worker thread inside async code** - `litellm.completion()` is a blocking call. Calling it directly inside an `async def` function would freeze the event loop for every other concurrent request. `asyncio.to_thread` runs it on a separate thread, keeping the loop free, the same reasoning applied to sync generators inside `StreamingResponse` back in Project 3.

**Two agent implementations kept side by side, deliberately, not one replacing the other** - `agent_service.py` (hand-built ReAct loop) and `langchain_agent_service.py` (LangChain's `create_agent`) both call the exact same `mcp_client_service` for tool discovery and execution, no duplication of the actual MCP logic. Comparing them directly showed both use the identical stopping condition (loop ends when no more tool calls are requested), confirmed from LangChain's own documentation rather than assumed.

**MCP tool results need both `content` and `structured_content` checked** - a tool that returns a plain dict without a declared output schema gets serialized into a text block inside `content`, not `structured_content`, which comes back `None`. Reading only `structured_content`, the shape shown in the SDK's own quickstart example, silently produced nothing.

**Reasoning text location is provider- and framework-specific, verified directly rather than assumed** - the hand-built agent's reasoning surfaced in `message.reasoning`, a field never mentioned in the tool-calling API documentation. LangChain's equivalent lives in `message.additional_kwargs["reasoning_content"]`, a different key entirely. Both were found only by printing the full, raw response object rather than trusting either framework's documented shape.

---

## Known Limitations, Named Directly

**Tool call batching is inconsistent, not architecturally limited** - The agent and multi-tool loops can correctly request several tool calls in a single response when the model chooses to, verified directly. But the same kind of request sometimes resolves across two or three separate round trips instead of one, adding latency and cost with no way to predict which will happen in advance. This is a real, unresolved production cost, not something the current architecture prevents.

**Tool execution has minimal resilience** - Retries treat every failure identically, transient and permanent. There is no timeout enforcement on a hanging tool call and no circuit breaking if an external dependency is consistently down. Planned for later, production-hardening work in this series.

**Re-ranking reorders but does not filter** - A weak, wrong-topic chunk that passes the RRF threshold still reaches the model as context after re-ranking, correctly pushed to the bottom of the list but never removed outright. The final answer has so far stayed correct because the model's own judgment ignores irrelevant context in the prompt, but that is a second, less controllable line of defense, not a guarantee.

**MCP servers are started manually, five separate terminals** - Functional for local development, not how this would run in any real deployment. A process manager or `docker-compose` setup is a natural next step, deliberately deferred rather than built today.

**Evaluate-then-act ordering is not enforced by either agent implementation** - tested directly and reproduced twice: a model given a conditional instruction ("if X takes longer than a week, do Y") correctly reasoned that the condition was false, but only after already deciding to act on Y, not before. Sequential reasoning at each step guarantees reasoning happens, not that it happens in the logically correct order relative to the actions it should gate. Neither the hand-built loop nor LangChain's `create_agent` has a structural mechanism that prevents this; a genuinely separate evaluation step before any action step would be required, which is exactly the kind of thing LangGraph's lower-level `StateGraph` is for, not attempted here.

**Neither agent framework detects hallucination or stalled progress as a distinct condition** - both `agent_service.py`'s hard iteration cap and LangGraph's `recursion_limit` (default 25, used by `create_agent`) only count total steps. A model stuck repeating the same failing tool call burns through every available step before either one stops, rather than being caught early. Confirmed as a known, unresolved gap across the broader agent framework ecosystem, not specific to this codebase.

---

## Environment Variables Reference

```bash
# Chat model
AI_MODEL=groq/openai/gpt-oss-20b
AI_API_KEY=your_api_key

# Embedding model
EMBEDDING_MODEL=ollama/nomic-embed-text
EMBEDDING_API_KEY=

# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_chat

# Conversation memory
MAX_CONVERSATION_MESSAGES=20

# RAG retrieval and filtering
RAG_MAX_CHUNKS=5
RAG_RRF_K=60
RAG_CANDIDATE_POOL_MULTIPLIER=4
RAG_RRF_THRESHOLD=0.02
RAG_SIMILARITY_THRESHOLD=0.55

# Re-ranking
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# MCP tool servers, must all be running for /agent, /agent-langchain, and /tool-chat
KNOWN_MCP_SERVERS=http://127.0.0.1:8001/mcp,http://127.0.0.1:8002/mcp,http://127.0.0.1:8003/mcp,http://127.0.0.1:8004/mcp,http://127.0.0.1:8005/mcp
```

Also requires, for the LangChain agent path:

```bash
pip install langchain langchain-groq
```