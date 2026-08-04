from __future__ import annotations

from typing import Any


def route_query(text: str) -> str:
    lowered = text.lower()
    predictive_keywords = ["predict", "break", "breakdown", "risk", "sensor", "last week", "incoming data"]
    return "predictor" if any(keyword in lowered for keyword in predictive_keywords) else "rag"


def build_rag_context(results: dict[str, Any]) -> str:
    documents = results.get("documents", []) or []
    if not documents or not documents[0]:
        return ""
    chunks = [chunk for chunk in documents[0] if chunk and str(chunk).strip()]
    return "\n\n".join(chunks)


def build_rag_prompt(query: str, context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful technical assistant that answers questions from an equipment manual. "
                "Use only the provided manual text. If the manual does not contain the answer, say so and ask the user to clarify. "
                "Keep the answer concise, friendly, and practical."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Manual excerpts:\n{context}\n\nQuestion: {query}\n\n"
                "If the exact procedure is not in the excerpts, explain that the manual does not provide a direct answer and suggest the user ask a more specific question."
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
