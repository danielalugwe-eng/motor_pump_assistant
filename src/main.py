from __future__ import annotations

from io import BytesIO
import os
from typing import Any
from typing import Literal

import numpy as np
import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency guard
    OpenAI = None

from src.data_utils import (
    build_breakdown_signal,
    extract_vibration_features_from_npz,
    load_latest_sensor_data,
)
from src.rag.llm import build_rag_context, build_rag_prompt, extract_answer_text, route_query
from src.rag.search import search_manual

load_dotenv()

app = FastAPI(title="Motor Pump Predictive System")
client = None
if OpenAI is not None:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key)
clf = None
if os.path.exists("models/best_fault_classifier.pkl"):
    try:
        clf = joblib.load("models/best_fault_classifier.pkl")
    except Exception:  # pragma: no cover - graceful fallback if model is invalid
        clf = None

SELECTED_FEATURES = ["rms", "kurtosis", "crest_factor", "dominant_freq", "spectral_energy", "spectral_entropy"]


def _build_sources(results: dict[str, Any], max_sources: int = 4) -> list[str]:
    sources: list[str] = []
    metadatas = results.get("metadatas", []) or []
    metadata_list = metadatas[0] if metadatas else []
    documents = results.get("documents", []) or []
    doc_list = documents[0] if documents else []
    for idx, _doc in enumerate(doc_list[:max_sources]):
        meta = metadata_list[idx] if idx < len(metadata_list) else {}
        page = meta.get("page") if isinstance(meta, dict) else None
        source = meta.get("source") if isinstance(meta, dict) else None
        if source or page:
            sources.append(f"{source or 'manual'} page {page or '?'}")
    return sources


def _source_quality_from_count(source_count: int) -> str:
    if source_count >= 3:
        return "high"
    if source_count == 2:
        return "medium"
    if source_count == 1:
        return "low"
    return "none"


class Query(BaseModel):
    text: str
    answer_style: Literal["short", "detailed"] = "detailed"


class SensorPayload(BaseModel):
    rms: float
    kurtosis: float
    crest_factor: float
    dominant_freq: float
    spectral_energy: float
    spectral_entropy: float


def predict_from_feature_row(feature_row: dict[str, Any]) -> dict[str, Any]:
    values = [float(feature_row[name]) for name in SELECTED_FEATURES]
    if clf is not None:
        try:
            proba = clf.predict_proba([values])[0]
            label = clf.classes_[int(proba.argmax())]
            confidence = float(proba.max())
            return {
                "route": "predictor",
                "answer": f"Predicted condition: {label} (confidence {confidence:.2f}).",
                "prediction": label,
                "confidence": confidence,
            }
        except Exception:
            return {
                "route": "predictor",
                "answer": "The fault model could not score the supplied sensor features. Please verify all values and try again.",
            }

    risk_score = float(feature_row["rms"] * 0.4 + feature_row["kurtosis"] * 0.6)
    status = "high_risk" if risk_score > 1.0 else "normal"
    return {
        "route": "predictor",
        "answer": f"Estimated status: {status} with risk score {risk_score:.2f} (fallback rule model).",
        "prediction": status,
        "confidence": None,
    }


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
        results = search_manual(text, k=6)
        context = build_rag_context(results)
        if not context:
            return {"route": "rag", "answer": "I could not find relevant text in the equipment manual."}

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"route": "rag", "answer": f"I found relevant manual excerpts, but the OpenAI API key is not configured. Context preview: {context[:800]}"}

        prompt = build_rag_prompt(text, context, answer_style=q.answer_style)
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=prompt,
            )
            answer = extract_answer_text(response)
            if not answer:
                answer = "I found relevant manual excerpts, but the LLM did not return a clear answer. Please try a more specific question."

            sources = _build_sources(results)
            source_count = len(sources)
            source_quality = _source_quality_from_count(source_count)

            return {
                "route": "rag",
                "answer": answer,
                "context_preview": context[:1000],
                "sources": sources,
                "source_count": source_count,
                "source_quality": source_quality,
                "answer_style": q.answer_style,
            }
        except Exception as llm_exc:  # pragma: no cover - graceful fallback
            # Retrieval-only fallback so user still gets useful grounded output.
            sources = _build_sources(results)
            source_count = len(sources)
            source_quality = _source_quality_from_count(source_count)
            preview = context[:500].strip()
            return {
                "route": "rag",
                "answer": (
                    "I found relevant manual context, but the LLM request failed. "
                    "Here is the closest manual evidence I found: "
                    f"{preview}"
                ),
                "context_preview": context[:1000],
                "sources": sources,
                "source_count": source_count,
                "source_quality": source_quality,
                "answer_style": q.answer_style,
            }
    except RuntimeError:
        return {
            "route": "rag",
            "answer": (
                "I could not complete manual retrieval right now, but I can still help. "
                "Please try the question again or ask with a specific component/procedure name "
                "(for example: fuse replacement, screw pump removal, or electrical panel access)."
            ),
            "source_quality": "none",
            "source_count": 0,
            "answer_style": q.answer_style,
        }
    except Exception:  # pragma: no cover - defensive fallback
        return {
            "route": "rag",
            "answer": (
                "I could not finish processing that manual question this time. "
                "Please retry, or ask a more specific version and I will answer from the manual evidence."
            ),
            "source_quality": "none",
            "source_count": 0,
            "answer_style": q.answer_style,
        }


@app.post("/predict")
def predict(sensor: SensorPayload) -> dict[str, Any]:
    feature_row = sensor.model_dump()
    return predict_from_feature_row(feature_row)


@app.post("/predict_raw")
async def predict_raw(file: UploadFile = File(...), signal_key: str = Form("DE")) -> dict[str, Any]:
    if not file.filename.lower().endswith(".npz"):
        return {"route": "predictor", "answer": "Please upload a raw CWRU .npz vibration file."}

    try:
        npz_bytes = await file.read()
        feature_row = extract_vibration_features_from_npz(npz_bytes, signal_key=signal_key)
        prediction = predict_from_feature_row(feature_row)
        return {
            **prediction,
            "source_file": file.filename,
            "signal_key": feature_row.get("signal_key"),
            "window_count": feature_row.get("window_count"),
            "raw_signal_length": feature_row.get("raw_signal_length"),
            "features": {name: feature_row.get(name) for name in SELECTED_FEATURES},
        }
    except Exception as exc:
        return {"route": "predictor", "answer": f"Could not process the raw vibration file: {exc}"}
