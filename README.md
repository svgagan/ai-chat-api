# Model-Agnostic AI Chat API

A production-structured AI Chat API built with FastAPI and LiteLLM.
Switch AI providers by changing two lines. Zero application code changes.

## Tech Stack
- Python, FastAPI, LiteLLM, Groq

## Setup

# Clone the repo
git clone <your-repo-url>
cd ai-chat-api

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your API key to .env

# Run
uvicorn app.main:app --reload

## API
- POST /api/v1/chat
- POST /api/v1/explain

## Design Decisions
- Service Layer pattern isolates AI logic from HTTP layer
- LiteLLM enables provider switching via config only
- Environment variables for all secrets
