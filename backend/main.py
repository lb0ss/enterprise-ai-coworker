import os
import uuid
from typing import Generator

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import chromadb
from openai import OpenAI

load_dotenv()

app = FastAPI(title="Enterprise AI Coworker")

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


class ChatRequest(BaseModel):
    question: str
    n_results: int = 5


def stream_answer(question: str, context_chunks: list[str]) -> Generator:
    context = "\n\n".join(context_chunks)
    prompt = f"""Use the following context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}
"""
    stream = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
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

    results = collection.query(
        query_embeddings=[question_vector],
        n_results=min(request.n_results, collection.count()),
    )
    context_chunks = results["documents"][0]

    return StreamingResponse(
        stream_answer(request.question, context_chunks),
        media_type="text/event-stream",
    )
