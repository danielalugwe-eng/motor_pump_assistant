from __future__ import annotations

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()


def ingest_manual(pdf_path: str | Path | None = None, collection_name: str = "manual_chunks") -> int:
    pdf_path = Path(pdf_path or Path("data/top_ex.pdf"))
    if not pdf_path.exists():
        raise FileNotFoundError(f"Manual PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

    chroma_client = chromadb.PersistentClient(path="data/chroma_db")
    collection = chroma_client.get_or_create_collection(collection_name)

    all_chunks: list[str] = []
    metadatas: list[dict[str, object]] = []
    ids: list[str] = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        chunks = splitter.split_text(text)
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadatas.append({"source": pdf_path.name, "page": page_num + 1})
            ids.append(f"{pdf_path.stem}_{page_num + 1}_{idx}")

    if not all_chunks:
        raise ValueError("No text could be extracted from the PDF")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: OPENAI_API_KEY not found. Storing chunks without embeddings.")
        # Store without embeddings as a fallback
        collection.add(documents=all_chunks, metadatas=metadatas, ids=ids)
        return len(all_chunks)

    client = OpenAI(api_key=api_key)
    batch_size = 10
    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start : start + batch_size]
        batch_ids = ids[start : start + batch_size]
        batch_metadatas = metadatas[start : start + batch_size]
        try:
            embeddings = [
                client.embeddings.create(model="text-embedding-3-small", input=chunk).data[0].embedding
                for chunk in batch
            ]
            collection.add(embeddings=embeddings, documents=batch, metadatas=batch_metadatas, ids=batch_ids)
        except Exception as exc:
            print(f"ERROR: Failed to embed batch {start}-{start + batch_size}: {exc}")
            print("Please check your network connection and OpenAI API key.")
            raise
    return len(all_chunks)


if __name__ == "__main__":
    print(ingest_manual())
