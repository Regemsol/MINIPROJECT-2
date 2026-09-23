from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "knn_ckd.pkl"

# Features used by the trained KNN model
FEATURES = ["hemo", "pcv", "sg", "htn", "rbcc"]

# Example patient values
patient = pd.DataFrame([
    {
        "hemo": 10.8,
        "pcv": 33.0,
        "sg": 1.015,
        "htn": 1,
        "rbcc": 3.8,
    }
], columns=FEATURES)

model = joblib.load(MODEL_PATH)
prediction = model.predict(patient)[0]
label = "CKD" if prediction == 1 else "Not CKD"

print(f"Patient prediction: {label}")
