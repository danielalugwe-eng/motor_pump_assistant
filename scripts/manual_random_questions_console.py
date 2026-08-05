import random
import requests

qs = [
    "Where are the fuses located?",
    "Can I perform an electronic modification on this equipment?",
    "How do I replace a fuse?",
    "What should I do if fuses keep blowing?",
    "How do I remove the screw pump?",
    "How do I reinsert the screw pump?",
    "What warning does the manual give before screw pump removal?",
    "Which panel should I open to access electrical components?",
]

random.shuffle(qs)
fails = 0
print("running", len(qs), "questions")

for q in qs:
    try:
        r = requests.post("http://127.0.0.1:8000/ask", json={"text": q, "answer_style": "detailed"}, timeout=90)
        d = r.json()
        a = (d.get("answer") or "").strip()
        bad = (
            r.status_code != 200
            or not a
            or "The RAG pipeline could not complete" in a
            or "LLM request failed" in a
        )
        print("---")
        print("Q:", q)
        print("status", r.status_code, "route", d.get("route"), "srcQ", d.get("source_quality"), "srcN", d.get("source_count"))
        print("A:", a[:220].replace("\n", " "))
        if bad:
            print("FAIL")
            fails += 1
    except Exception as exc:
        print("---")
        print("Q:", q)
        print("EXCEPTION", exc)
        print("FAIL")
        fails += 1

print("TOTAL_FAILS", fails)
