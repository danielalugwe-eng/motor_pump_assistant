import json
import requests

q = "can I perform an electronic modification on this equipment"
r = requests.post(
    "http://127.0.0.1:8000/ask",
    json={"text": q, "answer_style": "detailed"},
    timeout=60,
)
with open("scripts/last_question_response.json", "w", encoding="utf-8") as f:
    json.dump({"status": r.status_code, "body": r.json() if r.headers.get('content-type','').startswith('application/json') else r.text}, f, indent=2)
print("saved")
