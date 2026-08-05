from __future__ import annotations

from typing import Any


PREDICTIVE_KEYWORDS = {
    "predict",
    "prediction",
    "break",
    "breakdown",
    "fault",
    "failure",
    "risk",
    "rul",
    "remaining life",
    "remaining useful life",
    "sensor",
    "signal",
    "vibration",
    "incoming data",
    "trend",
    "anomaly",
    "health score",
    "condition monitoring",
}

MANUAL_KEYWORDS = {
    "manual",
    "procedure",
    "step",
    "replace",
    "change",
    "remove",
    "install",
    "fuse",
    "pump",
    "electrical",
    "electronic",
    "modification",
    "modify",
    "can i",
    "allowed",
    "permit",
    "permission",
    "compliance",
    "regulation",
    "wiring",
    "cable",
    "terminal",
    "safety",
    "warning",
    "maintenance",
    "clean",
    "calibrate",
    "panel",
    "component",
}


def route_query(text: str) -> str:
    lowered = text.lower().strip()

    # score each intent to make routing less brittle than one-off substring checks
    predictive_score = sum(1 for keyword in PREDICTIVE_KEYWORDS if keyword in lowered)
    manual_score = sum(1 for keyword in MANUAL_KEYWORDS if keyword in lowered)

    if predictive_score > manual_score and predictive_score > 0:
        return "predictor"
    return "rag"


def build_rag_context(results: dict[str, Any]) -> str:
    # Results are expected to contain parallel lists under keys 'documents' and 'metadatas'
    documents_list = results.get("documents", []) or []
    metadatas_list = results.get("metadatas", []) or []

    if not documents_list or not documents_list[0]:
        return ""

    docs = documents_list[0]
    metas = metadatas_list[0] if metadatas_list and metadatas_list[0] else [None] * len(docs)

    paired = []
    for doc, meta in zip(docs, metas):
        if not doc or not str(doc).strip():
            continue
        file_order = meta.get("file_order") if isinstance(meta, dict) else None
        page = meta.get("page") if isinstance(meta, dict) else None
        chunk = meta.get("chunk") if isinstance(meta, dict) else None
        source = meta.get("source") if isinstance(meta, dict) else "manual"
        paired.append({
            "doc": doc,
            "meta": meta or {},
            "source": source,
            "file_order": file_order or 0,
            "page": page or 0,
            "chunk": chunk or 0,
        })

    # Preserve the simple original behavior when the search result has no useful metadata.
    if not any(item["meta"] for item in paired):
        return "\n\n".join(item["doc"] for item in paired)

    # sort by file, page, chunk for source-level ordering
    paired.sort(key=lambda x: (x["source"], x["file_order"], x["page"], x["chunk"]))

    ordered_chunks: list[str] = []
    for item in paired:
        source = item["source"]
        page = item["page"]
        title = f"[{source} page {page}]"
        ordered_chunks.append(f"{title}\n{item['doc']}")

    return "\n\n".join(ordered_chunks)


def build_rag_prompt(query: str, context: str, answer_style: str = "detailed") -> list[dict[str, str]]:
    style_instruction = (
        "Keep the response short: 2-4 sentences, one key source reference, and no extra detail."
        if answer_style == "short"
        else "Give a detailed response with clear steps when procedures are present and include source references."
    )

    return [
        {
            "role": "system",
            "content": (
                "You are a careful technical assistant that answers questions using only the provided equipment manual excerpts. "
                "Be direct, practical, and helpful. Do not invent facts outside the excerpts. "
                "If relevant text exists, answer clearly in plain language and include source references like [source page X]. "
                "If the exact answer is missing, still provide the closest supported guidance from the excerpts and clearly label assumptions. "
                "Only ask a follow-up question when the user's request is ambiguous or impossible to answer from the provided text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Manual excerpts:\n{context}\n\nQuestion: {query}\n\n"
                "Instructions:\n"
                "1) Use only the provided excerpts.\n"
                "2) If the answer exists, state it directly first.\n"
                "3) Add 1-2 bullet steps if there is a procedure.\n"
                "4) Cite sources in brackets, e.g. [top_ex.pdf page 39].\n"
                "5) If details are partial, say what is known and what is missing, then ask one clarifying follow-up.\n"
                f"6) Style: {style_instruction}"
            ),
        },
    ]


def extract_answer_text(response: Any) -> str:
    if response is None:
        return ""

    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        message = getattr(choice, "message", None)
        if isinstance(message, dict):
            return message.get("content", "").strip()
        if message is not None:
            return getattr(message, "content", "").strip()

    if hasattr(response, "output") and response.output:
        output_item = response.output[0]
        if isinstance(output_item, dict):
            return output_item.get("content", "").strip()
        return str(output_item).strip()

    return ""
