import json
import random
import requests

URL = "http://127.0.0.1:8000/ask"

QUESTIONS = [
    "Where are the fuses located?",
    "Can I perform an electronic modification on this equipment?",
    "How do I replace a fuse?",
    "What should I do if fuses keep blowing?",
    "How do I remove the screw pump?",
    "How do I reinsert the screw pump?",
    "What warning does the manual give before screw pump removal?",
    "Which panel should I open to access electrical components?",
    "Is there guidance about auxiliary 24V and 230V fuses?",
    "What maintenance section covers screw pump procedures?",
    "What should be checked after replacing a fuse?",
    "Can I modify wiring without manufacturer instructions?",
]

random.shuffle(QUESTIONS)

results = []
failures = []
for q in QUESTIONS:
    row = {"question": q}
    try:
        resp = requests.post(URL, json={"text": q, "answer_style": "detailed"}, timeout=90)
        row["status"] = resp.status_code
        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"raw": resp.text}
        row["answer"] = data.get("answer", "") if isinstance(data, dict) else str(data)
        row["route"] = data.get("route") if isinstance(data, dict) else None
        row["source_count"] = data.get("source_count") if isinstance(data, dict) else None
        row["source_quality"] = data.get("source_quality") if isinstance(data, dict) else None
        row["has_error_text"] = "error" in row["answer"].lower() if row["answer"] else True

        if resp.status_code != 200:
            failures.append({"question": q, "reason": f"http_{resp.status_code}"})
        elif not row["answer"].strip():
            failures.append({"question": q, "reason": "empty_answer"})
        elif "The RAG pipeline could not complete" in row["answer"]:
            failures.append({"question": q, "reason": "rag_pipeline_error"})
        elif "LLM request failed" in row["answer"]:
            failures.append({"question": q, "reason": "llm_failed"})
    except Exception as exc:
        row["status"] = None
        row["answer"] = str(exc)
        failures.append({"question": q, "reason": f"exception: {exc}"})

    results.append(row)

summary = {
    "total": len(QUESTIONS),
    "failures": len(failures),
    "failure_items": failures,
    "results": results,
}

with open("scripts/manual_random_questions_results.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps({"total": summary["total"], "failures": summary["failures"]}, indent=2))
