from __future__ import annotations

import os
from typing import Any

import joblib
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

from src.data_utils import build_breakdown_signal, load_latest_sensor_data
from src.rag.llm import build_rag_context, build_rag_prompt, extract_answer_text, route_query
from src.rag.search import search_manual

load_dotenv()

app = FastAPI(title="Motor Pump Predictive System")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
clf = joblib.load("models/best_fault_classifier.pkl") if os.path.exists("models/best_fault_classifier.pkl") else None


class Query(BaseModel):
    text: str


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Motor Pump Predictive System API is running."}


@app.post("/ask")
def ask(q: Query) -> dict[str, Any]:
    text = q.text
    if route_query(text) == "predictor":
        try:
            data = load_latest_sensor_data()
            features = data[["rms", "kurtosis"]].copy()
            signal = build_breakdown_signal(features)
            return {"route": "predictor", "answer": f"Incoming data suggests {signal['status']} with risk score {signal['risk_score']:.2f}."}
        except FileNotFoundError:
            return {"route": "predictor", "answer": "Incoming sensor data file is not available yet."}

    try:
        results = search_manual(text)
        context = build_rag_context(results)
        if not context:
            return {"route": "rag", "answer": "I could not find relevant text in the equipment manual."}

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"route": "rag", "answer": f"I found relevant manual excerpts, but the OpenAI API key is not configured. Context preview: {context[:800]}"}

        prompt = build_rag_prompt(text, context)
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=prompt,
            )
            answer = extract_answer_text(response)
            if not answer:
                answer = "I found relevant manual excerpts, but the LLM did not return a clear answer. Please try a more specific question."

            sources = []
            metadatas = results.get("metadatas", []) or []
            metadata_list = metadatas[0] if metadatas else []
            for idx, doc in enumerate(results["documents"][0][:3]):
                meta = metadata_list[idx] if idx < len(metadata_list) else {}
                page = meta.get("page") if isinstance(meta, dict) else None
                source = meta.get("source") if isinstance(meta, dict) else None
                if source or page:
                    sources.append(f"{source or 'manual'} page {page or '?'}")

            return {
                "route": "rag",
                "answer": answer,
                "context_preview": context[:1000],
                "sources": sources,
            }
        except Exception as llm_exc:  # pragma: no cover - graceful fallback
            return {
                "route": "rag",
                "answer": (
                    "I found relevant manual context, but the LLM request failed. "
                    "Please check your OpenAI API settings or try again."
                ),
                "context_preview": context[:1000],
            }
    except RuntimeError as exc:
        return {"route": "rag", "answer": str(exc)}
    except Exception as exc:  # pragma: no cover - defensive fallback
        return {"route": "rag", "answer": f"The RAG pipeline could not complete: {exc}"}
