import json
import os
import time
import uuid
from typing import AsyncGenerator, Generator

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, ToolMessage
from langsmith import traceable
from pydantic import BaseModel

import chromadb
from openai import OpenAI

load_dotenv()

from agent import get_agent
from analytics import get_analytics, init_db, log_feedback, log_query
from cache import get_cached_answer, set_cached_answer

app = FastAPI(title="Enterprise AI Coworker")

init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunks.append(" ".join(words[start:end]))
        start = end - CHUNK_OVERLAP
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug/chunks")
def debug_chunks():
    return collection.get()


@app.get("/documents")
def get_documents():
    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas", [])

    # aggregate chunk counts per filename
    counts: dict[str, int] = {}
    for m in metadatas:
        filename = m.get("filename", "unknown")
        counts[filename] = counts.get(filename, 0) + 1

    return {
        "documents": [{"filename": f, "chunks_indexed": c} for f, c in counts.items()],
        "total_chunks": len(metadatas),
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith((".txt", ".md", ".py", ".pdf")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    raw = await file.read()

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Could not decode file as UTF-8")

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="File appears to be empty")

    embeddings = embed_texts(chunks)

    ids = [str(uuid.uuid4()) for _ in chunks]
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"filename": file.filename, "chunk_index": i} for i, _ in enumerate(chunks)],
    )

    return {"filename": file.filename, "chunks_indexed": len(chunks)}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []
    n_results: int = 5


@traceable(name="chat", run_type="chain")
def stream_answer(question: str, context_chunks: list[str], history: list[ChatMessage]) -> Generator:
    context = "\n\n".join(context_chunks)
    system_prompt = f"""You are a helpful assistant that answers questions based on the provided document context.
If the answer is not in the context, say you don't know.

Context:
{context}"""

    messages = [{"role": "system", "content": system_prompt}]
    messages += [{"role": m.role, "content": m.content} for m in history]
    messages.append({"role": "user", "content": question})

    stream = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield f"data: {token}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/chat")
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # embed_texts returns a list of vectors (one per input); [0] unwraps the single question vector
    question_vector = embed_texts([request.question])[0]

    # check Redis for a semantically similar cached answer before calling OpenAI
    start_ms = int(time.time() * 1000)

    cached = get_cached_answer(question_vector)
    if cached:
        latency_ms = int(time.time() * 1000) - start_ms
        query_id = log_query(request.question, cached, latency_ms, 0, cache_hit=True)

        def stream_cached():
            yield f"data: {cached}\n\n"
            yield f"data: [QUERY_ID:{query_id}]\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_cached(), media_type="text/event-stream")

    results = collection.query(
        query_embeddings=[question_vector],
        n_results=min(request.n_results, collection.count()),
    )
    context_chunks = results["documents"][0]

    def stream_and_cache():
        full_answer = []
        for chunk in stream_answer(request.question, context_chunks, request.history):
            if chunk != "data: [DONE]\n\n":
                token = chunk[6:].rstrip("\n")
                full_answer.append(token)
            yield chunk

        answer = "".join(full_answer)
        latency_ms = int(time.time() * 1000) - start_ms
        set_cached_answer(question_vector, answer)
        query_id = log_query(request.question, answer, latency_ms, 0, cache_hit=False)
        yield f"data: [QUERY_ID:{query_id}]\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_and_cache(), media_type="text/event-stream")


class AgentRequest(BaseModel):
    task: str


async def stream_agent_events(task: str) -> AsyncGenerator:
    inputs = {"messages": [{"role": "user", "content": task}]}

    # MCP client lifecycle is tied to the request — connects to mcp_server.py via stdio
    agent = await get_agent()

    # astream_events streams every internal ReAct loop event as it happens
    async for event in agent.astream_events(inputs, version="v2"):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            if token:
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        elif kind == "on_tool_start":
            yield f"data: {json.dumps({'type': 'tool_call', 'tool': event['name'], 'input': str(event['data'].get('input', ''))})}\n\n"

        elif kind == "on_tool_end":
            yield f"data: {json.dumps({'type': 'tool_result', 'tool': event['name'], 'content': str(event['data'].get('output', ''))[:300]})}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/agent")
async def run_agent(request: AgentRequest):
    if not request.task.strip():
        raise HTTPException(status_code=400, detail="Task cannot be empty")

    return StreamingResponse(
        stream_agent_events(request.task),
        media_type="text/event-stream",
    )


class FeedbackRequest(BaseModel):
    query_id: int
    rating: int  # 1 = thumbs up, -1 = thumbs down


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    if request.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="Rating must be 1 or -1")
    log_feedback(request.query_id, request.rating)
    return {"status": "ok"}


@app.get("/analytics")
def analytics():
    return get_analytics()
