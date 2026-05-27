from pathlib import Path

import pandas as pd

from src.predict import EmailClassifierPredictor
from src.preprocessing import clean_text


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

print("Loading predictor...")
predictor = EmailClassifierPredictor(MODEL_DIR)
print(f"Loaded model from: {predictor.model_path}")

samples = [
    "Please review the attached invoice for last month.",
    "Don't miss our weekly deals and offers!",
    "Win big with our new promotion.",
    "Lunch this weekend?",
    "Urgent: Update your password to keep your account secure.",
]

rows = []
for s in samples:
    res = predictor.predict_text(s)
    rows.append({
        "text": s,
        "cleaned_email": clean_text(s),
        "prediction": res.label,
        "confidence": res.confidence,
    })

out = pd.DataFrame(rows)
print(out.to_string(index=False))

out_csv = BASE_DIR / "data" / "sample_predictions.csv"
out_xlsx = BASE_DIR / "data" / "sample_predictions.xlsx"
out.to_csv(out_csv, index=False)
out.to_excel(out_xlsx, index=False)
print(f"Wrote sample predictions to: {out_csv} and {out_xlsx}")
