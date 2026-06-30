# Model-Agnostic AI Chat API

A production-structured AI API built with FastAPI and LiteLLM.
Switch AI providers by changing two lines. Zero application code changes.

## Tech Stack
- Python, FastAPI, LiteLLM, Instructor, Groq

## Setup

### Clone the repo
git clone <your-repo-url>
cd ai-chat-api

#### Create virtual environment
python -m venv venv
source venv/bin/activate

### Install dependencies
pip install -r requirements.txt

### Configure environment
cp .env.example .env
### Add your API key to .env

### Run
uvicorn app.main:app --reload

## API

POST /api/v1/chat
General purpose chat with a configurable system prompt.

POST /api/v1/explain
Explains any topic tailored to a given audience (e.g. "explain black holes to a 5 year old").
System prompt is built internally — caller never controls AI behavior directly.

POST /api/v1/extract
Extracts structured data from unstructured customer support text
(name, age, order number, email, issue, sentiment) using schema-enforced
structured output. Returns a validated JSON object, not free text.

## Design Decisions

- Service Layer pattern isolates AI logic from HTTP layer
- LiteLLM enables provider switching via config only
- Environment variables for all secrets, never hardcoded
- Callers never control system prompts or model behavior directly —
  these are business decisions, not user-controllable inputs (prevents
  prompt injection)
- Structured output (extract endpoint) uses Pydantic schema enforcement
  via Instructor, not just prompt-based JSON requests — guarantees shape,
  with automatic retry-with-feedback if the AI's output fails validation
- All extracted fields are Optional by design, since real-world unstructured
  text rarely contains every possible field — required fields would cause
  valid customer messages to fail