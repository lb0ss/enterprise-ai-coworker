import os

import chromadb
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "workspace")

mcp = FastMCP("DocMind Tools")


@mcp.tool()
def search_documents(query: str) -> str:
    """Search the indexed documents for content relevant to the query."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[query],
    )
    query_vector = response.data[0].embedding

    count = collection.count()
    if count == 0:
        return "No documents have been indexed yet."

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(5, count),
    )
    chunks = results["documents"][0]
    return "\n\n".join(chunks)


@mcp.tool()
def read_file(filename: str) -> str:
    """Read a file from the workspace directory. Only files inside the workspace are accessible."""
    safe_path = os.path.realpath(os.path.join(WORKSPACE_DIR, filename))

    # Prevent path traversal attacks (e.g. filename = "../../.env")
    if not safe_path.startswith(os.path.realpath(WORKSPACE_DIR)):
        return "Access denied: file is outside the workspace directory."

    if not os.path.exists(safe_path):
        return f"File not found: {filename}"

    with open(safe_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    mcp.run()
