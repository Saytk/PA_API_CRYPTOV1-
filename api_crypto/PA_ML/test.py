from pathlib import Path
import json

path = Path("crypto_forecast_ml/config/credentials.json")

with open(path, encoding="utf-8") as f:
    data = json.load(f)

print("✅ JSON is valid and UTF-8 encoded.")
