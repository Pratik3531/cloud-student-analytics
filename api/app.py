from fastapi import FastAPI
import xgboost as xgb
import pandas as pd
import numpy as np

app = FastAPI()

# ---------- Load Models ----------
reg_model = xgb.Booster()
clf_model = xgb.Booster()

reg_model.load_model("models/reg_model.xgb")
clf_model.load_model("models/clf_model.xgb")

# ---------- ROUTES ----------
@app.get("/")
def home():
    return {"message": "Student Analytics Cloud API is running!"}

@app.post("/predict")
def predict(data: dict):
    # Convert input to DataFrame
    df = pd.DataFrame([data])

    # XGBoost expects DMatrix
    dmatrix = xgb.DMatrix(df)

    # Predictions
    final_grade = float(reg_model.predict(dmatrix)[0])
    pass_prob = float(clf_model.predict(dmatrix)[0])
    pass_fail = int(pass_prob >= 0.5)

    # Stress score (dummy logic)
    stress_score = float(max(0, 1 - df["engagement_score"][0]/100))

    # Attendance anomaly (dummy logic)
    attendance_flag = int(df["absences"][0] > 10)

    return {
        "final_grade": round(final_grade, 2),
        "pass_probability": round(pass_prob, 2),
        "pass_fail": pass_fail,
        "stress_score": round(stress_score, 2),
        "attendance_anomaly": attendance_flag
    }
