from __future__ import annotations

import os
from typing import Any

import chromadb
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency guard
    OpenAI = None

load_dotenv()


def search_manual(query: str, k: int = 4, collection_name: str = "manual_chunks") -> dict[str, Any]:
    chroma_client = chromadb.PersistentClient(path="data/chroma_db")
    collection = chroma_client.get_or_create_collection(collection_name)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to the .env file first.")

    try:
        client = OpenAI(api_key=api_key)
        query_embedding = client.embeddings.create(model="text-embedding-3-small", input=query).data[0].embedding
    except Exception as exc:  # pragma: no cover - graceful fallback for missing API access
        raise RuntimeError(f"Unable to create embedding for manual search: {exc}") from exc

    return collection.query(query_embeddings=[query_embedding], n_results=k, include=["documents", "metadatas"])
