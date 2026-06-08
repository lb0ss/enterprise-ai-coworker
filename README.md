# DocMind — Enterprise AI Coworker

A full-stack AI coworker built across three phases: RAG pipeline, agentic workflows, and LLMOps. Upload documents, ask questions with streaming answers, run a ReAct agent with MCP-powered tools, and monitor everything through a live analytics dashboard.

![DocMind screenshot](screenshots/Screenshot%202026-06-07%20at%207.08.22%20PM.png)

---

## What it does

- **Chat** — upload a document, ask questions, get answers grounded in your content streamed token by token via SSE
- **Agent** — give the agent a task; it reasons, calls tools (semantic search, file read), observes results, and repeats until it has an answer
- **Analytics** — live LLMOps dashboard showing cache hit rate, avg latency, token usage, and thumbs up/down feedback per query

---

## Tech Stack

### Backend
| Layer | Technology |
|---|---|
| API server | FastAPI + Uvicorn |
| Vector database | ChromaDB |
| Embedding model | OpenAI text-embedding-3-small |
| LLM | OpenAI gpt-4o-mini |
| Agent framework | LangGraph (ReAct) |
| Tool protocol | MCP via FastMCP + langchain-mcp-adapters |
| Semantic cache | Redis (cosine similarity, threshold 0.92) |
| Observability | LangSmith tracing |
| Analytics | SQLite |
| Package manager | uv |

### Frontend
| Layer | Technology |
|---|---|
| Framework | React 19 + TypeScript |
| Build tool | Vite |
| Styling | Tailwind CSS v4 |
| Streaming | Native fetch + SSE |

---

## Architecture

### Phase 1 — RAG Pipeline
```
Upload:  File → Chunk (500w / 50 overlap) → Embed → ChromaDB
Chat:    Question → Embed → Retrieve top 5 → Augment prompt → Stream via SSE
```

### Phase 2 — ReAct Agent
```
Task → LLM reasons → picks MCP tool → tool executes → LLM observes → repeats → answer
```
Tools exposed via MCP: `search_documents` (semantic ChromaDB search), `read_file` (sandboxed to `workspace/`)

### Phase 3 — LLMOps
```
Every /chat request → check Redis cache → hit: stream cached answer
                                        → miss: call OpenAI → cache result → log to SQLite
LangSmith traces every /chat and /agent call end-to-end
[QUERY_ID:N] piggybacked on SSE stream → frontend attaches thumbs up/down feedback
```

---

## Running locally

### Prerequisites
- Python 3.11+, Node 20+, Redis running on port 6379

### Backend
```bash
cd backend
uv sync
uv run uvicorn main:app --reload --port 8001
```

`backend/.env` requires:
```
OPENAI_API_KEY=...
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=enterprise-ai-coworker
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

### Frontend
```bash
cd frontend
npm install
npm run dev -- --port 5173
```

---

## Key concepts

**RAG** · **ReAct** · **MCP** · **LangGraph** · **LangSmith** · **Semantic caching** · **SSE streaming** · **Vector embeddings** · **Agentic workflows** · **LLMOps**
