import json
import sys
from pathlib import Path
import warnings

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from src.main import app

warnings.filterwarnings("ignore")

client = TestClient(app)

raw_file = Path("data/raw/CWRU_Bearing_NumPy/Data/1750 RPM/1750_B_14_DE12.npz")

if raw_file.exists():
    with raw_file.open("rb") as f:
        r1 = client.post(
            "/predict_raw",
            files={"file": (raw_file.name, f, "application/octet-stream")},
            data={"signal_key": "DE"},
        )
    print("predict_raw status:", r1.status_code)
    try:
        print(json.dumps(r1.json(), indent=2)[:1800])
    except Exception:
        print(r1.text[:1800])
else:
    print("predict_raw file missing:", raw_file)

r2 = client.post("/ask", json={"text": "predict risk from incoming sensor data"})
print("ask status:", r2.status_code)
try:
    print(json.dumps(r2.json(), indent=2)[:1800])
except Exception:
    print(r2.text[:1800])
