import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from src.main import app


RAW_FILE = Path("data/raw/CWRU_Bearing_NumPy/Data/1750 RPM/1750_B_14_DE12.npz")
OUT_FILE = Path("scripts/e2e_test_result.json")


client = TestClient(app)

results = {}

# 1) raw vibration prediction
if RAW_FILE.exists():
    with RAW_FILE.open("rb") as f:
        resp = client.post(
            "/predict_raw",
            files={"file": (RAW_FILE.name, f, "application/octet-stream")},
            data={"signal_key": "DE"},
        )
    results["predict_raw_status"] = resp.status_code
    try:
        results["predict_raw_json"] = resp.json()
    except Exception:
        results["predict_raw_text"] = resp.text
else:
    results["predict_raw_error"] = f"Raw file not found: {RAW_FILE}"

# 2) chat/manual route
resp2 = client.post("/ask", json={"text": "Where are the fuses located?"})
results["ask_status"] = resp2.status_code
try:
    results["ask_json"] = resp2.json()
except Exception:
    results["ask_text"] = resp2.text

OUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(str(OUT_FILE))
