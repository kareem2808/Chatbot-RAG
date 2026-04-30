# UMKM RAG Assistant: Production-Ready AI Chatbot

A high-performance, modular AI chatbot backend built specifically for Small and Medium Enterprises (UMKM). This project implements an advanced Retrieval-Augmented Generation (RAG) architecture powered by FastAPI, ChromaDB, and multi-provider LLM support (Google Gemini & Groq) to deliver fast, accurate, and context-aware responses based on store data.

## Architecture Overview

The system is built on a robust, decoupling-first architecture to ensure scalability and reliability:

- **Intent Routing (Zero-Shot Classification)**: Before any retrieval occurs, a lightweight router evaluates the user's intent. If the query is out-of-domain (e.g., chitchat or irrelevant questions), the system immediately returns a fallback response. This saves tokens, reduces latency, and prevents hallucination on non-business topics.
- **Hybrid Retrieval (ChromaDB + BM25)**: Combines dense vector embeddings for semantic understanding with sparse keyword matching to ensure maximum retrieval accuracy across product catalogs, FAQs, and store policies.
- **Corrective RAG (CRAG)**: An evaluation layer that scores retrieved context chunks for relevance against the user's query. Irrelevant chunks are filtered out before being passed to the generation model, ensuring the final answer is grounded strictly in facts.
- **Multi-Provider LLM Engine**: Seamlessly switch between Google Gemini (free tier) and Groq (high-speed Llama 3) via environment variables, ensuring high availability and bypassing strict API rate limits.

## AI Engineering Approach: Precision Prompting & Efficiency

This project was built from the ground up using advanced AI Engineering and Prompt Engineering techniques to ensure an enterprise-grade codebase.

- **Rapid Prototyping via Prompt Engineering**: The entire architecture was conceptualized and executed using deliberate, step-by-step LLM interactions, moving from core abstractions to fully wired API endpoints.
- **JSON-Structured Prompting**: I utilized strict "JSON-Structured Prompting" to interface with the LLM during development. By forcing the AI to plan and respond in strict JSON formats, the development process was highly deterministic.
- **Hallucination Elimination & Cost Reduction**: This methodology eliminated inherent LLM hallucinations, enforced a strict modular architecture, and drastically reduced token consumption and API costs during the build phase.
- **Production-First Mindset**: The approach demonstrates a production-first mindset by forcing precise, industry-standard code outputs from the start—including built-in retry logic, exponential backoff, rate limiting, and centralized configuration management.

## Tech Stack

- **Backend Framework**: FastAPI (Python 3.10+)
- **Vector Database**: ChromaDB (Persistent local storage)
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **LLM Providers**: Google Gemini (`gemini-2.5-flash`), Groq (`llama-3.1-8b-instant`)
- **Frontend / UI**: Streamlit (Chat interface)
- **Validation & Config**: Pydantic, Pydantic-Settings

## Local Setup Instructions

Follow these steps to run the UMKM RAG Assistant locally:

### 1. Clone & Environment Setup
Clone the repository and create a virtual environment:
```bash
git clone <repository_url>
cd <repository_name>
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Note: Ensure `groq`, `google-generativeai`, `fastapi`, `chromadb`, and `streamlit` are installed as per your dependencies file).*

### 3. Configuration
Copy the environment template and fill in your API keys:
```bash
cp .env.example .env
```
Inside `.env`, configure your LLM provider:
```env
LLM_PROVIDER="groq"
GROQ_API_KEY="your_groq_api_key"
# Optional: GEMINI_API_KEY="your_gemini_key"
API_KEYS="dev-key-umkm-2024"  # For backend endpoint authentication
```

### 4. Data Seeding
Before chatting, ensure your dummy store data (`toko_sejahtera.txt`) has been ingested into ChromaDB.
```bash
python tests/seed_db.py
```

### 5. Start the Services
Run the backend API:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
In a new terminal window, start the Streamlit UI:
```bash
streamlit run src/ui/app.py
```

Navigate to `http://localhost:8501` to test the chatbot!
