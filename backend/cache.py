import json
import os

import numpy as np
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
SIMILARITY_THRESHOLD = 0.92  # cosine similarity score above which we consider a cache hit
CACHE_TTL = 3600  # cached answers expire after 1 hour

redis_client = redis.from_url(REDIS_URL, decode_responses=True)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


def get_cached_answer(question_vector: list[float]) -> str | None:
    """Search Redis for a cached answer whose question vector is similar enough."""
    keys = redis_client.keys("cache:*")

    for key in keys:
        entry = redis_client.get(key)
        if not entry:
            continue

        data = json.loads(entry)
        cached_vector = data["vector"]
        similarity = cosine_similarity(question_vector, cached_vector)

        if similarity >= SIMILARITY_THRESHOLD:
            print(f"[CACHE HIT] similarity: {similarity:.4f} — returning cached answer")
            return data["answer"]
        else:
            print(f"[CACHE MISS] similarity: {similarity:.4f} — below threshold {SIMILARITY_THRESHOLD}")

    print("[CACHE MISS] no entries in Redis yet")
    return None


def set_cached_answer(question_vector: list[float], answer: str) -> None:
    """Store a question vector and its answer in Redis with a TTL."""
    key = f"cache:{hash(tuple(question_vector))}"
    entry = json.dumps({"vector": question_vector, "answer": answer})
    redis_client.setex(key, CACHE_TTL, entry)
