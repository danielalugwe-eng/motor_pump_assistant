from __future__ import annotations

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()


def ingest_manual(
    pdf_path: str | Path | None = None,
    collection_name: str = "manual_chunks",
    skip_embeddings: bool = False,
) -> int:
    """Ingest a single PDF file or all PDFs in a directory.

    If `pdf_path` is a directory, PDFs are ingested in sorted filename order
    and a `file_order` is assigned to preserve reading order across multiple
    documents. Each chunk metadata includes: source (filename), page, chunk,
    and file_order.
    """
    pdf_path = Path(pdf_path or Path("data/top_ex.pdf"))
    if not pdf_path.exists():
        raise FileNotFoundError(f"Manual PDF not found: {pdf_path}")

    # allow ingesting a directory of PDFs in deterministic order
    pdf_files: list[Path] = []
    if pdf_path.is_dir():
        pdf_files = sorted([p for p in pdf_path.glob("*.pdf")])
    else:
        pdf_files = [pdf_path]

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chroma_client = chromadb.PersistentClient(path="data/chroma_db")

    if skip_embeddings:
        # Recreate a fresh collection without an embedding function when storing text-only chunks.
        existing_collections = [col.name for col in chroma_client.list_collections()]
        if collection_name in existing_collections:
            chroma_client.delete_collection(collection_name)
        collection = chroma_client.create_collection(collection_name, embedding_function=None)
    else:
        collection = chroma_client.get_or_create_collection(collection_name, embedding_function=None)

    all_chunks: list[str] = []
    metadatas: list[dict[str, object]] = []
    ids: list[str] = []

    for file_order, file_path in enumerate(pdf_files, start=1):
        reader = PdfReader(str(file_path))
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            chunks = splitter.split_text(text)
            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadatas.append(
                    {
                        "source": file_path.name,
                        "page": page_num + 1,
                        "chunk": idx,
                        "file_order": file_order,
                    }
                )
                ids.append(f"{file_path.stem}_{page_num + 1}_{idx}")

    if not all_chunks:
        raise ValueError("No text could be extracted from the PDF")

    api_key = os.getenv("OPENAI_API_KEY")
    if skip_embeddings or not api_key:
        if not api_key and not skip_embeddings:
            print("WARNING: OPENAI_API_KEY not found. Storing chunks without embeddings.")
        elif skip_embeddings:
            print("INFO: Skipping embeddings as requested. Storing chunks only.")
        collection.add(documents=all_chunks, metadatas=metadatas, ids=ids)
        return len(all_chunks)

    client = OpenAI(api_key=api_key, timeout=20, max_retries=1)
    batch_size = 10
    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start : start + batch_size]
        batch_ids = ids[start : start + batch_size]
        batch_metadatas = metadatas[start : start + batch_size]
        try:
            embeddings = [
                client.embeddings.create(model="text-embedding-3-small", input=chunk, timeout=20).data[0].embedding
                for chunk in batch
            ]
            collection.add(embeddings=embeddings, documents=batch, metadatas=batch_metadatas, ids=batch_ids)
        except Exception as exc:
            print(f"ERROR: Failed to embed batch {start}-{start + batch_size}: {exc}")
            print("Falling back to storing chunks without embeddings. Retry ingestion after fixing OpenAI access.")
            collection.add(documents=batch, metadatas=batch_metadatas, ids=batch_ids)
    return len(all_chunks)


if __name__ == "__main__":
    print(ingest_manual())
