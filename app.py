import gradio as gr
import joblib
import pandas as pd
import numpy as np

# Load model and columns
model = joblib.load("xgb_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Threshold from notebook
BEST_THRESHOLD = 0.89


def predict_fraud(*args):

    input_data = {}

    input_data["Time"] = args[0]

    for i in range(1, 29):
        input_data[f"V{i}"] = args[i]

    amount = args[29]

    # Simple scaling approximation
    # (avoids requiring creditcard.csv)
    scaled_amount = (amount - 88.35) / 250.12

    input_data["scaled_amount"] = scaled_amount

    df = pd.DataFrame([input_data])

    df = df[model_columns]

    probability = model.predict_proba(df)[:, 1][0]

    prediction = int(probability > BEST_THRESHOLD)

    if prediction == 1:
        return {
            "Fraudulent Transaction": float(probability),
            "Legitimate Transaction": float(1 - probability)
        }
    else:
        return {
            "Legitimate Transaction": float(1 - probability),
            "Fraudulent Transaction": float(probability)
        }


inputs = [
    gr.Number(label="Time"),
    *[gr.Number(label=f"V{i}") for i in range(1, 29)],
    gr.Number(label="Amount")
]

demo = gr.Interface(
    fn=predict_fraud,
    inputs=inputs,
    outputs=gr.Label(num_top_classes=2),
    title="Credit Card Fraud Detection",
    description="Predict whether a transaction is fraudulent using an XGBoost model."
)

demo.launch()