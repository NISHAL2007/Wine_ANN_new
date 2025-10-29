import gradio as gr
import joblib
import numpy as np

model = joblib.load("wine_ann_model.joblib")
scaler = joblib.load("wine_scaler.joblib")

feature_names = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
    "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
]

def predict_wine_quality(*inputs):
    features = np.array(inputs).reshape(1, -1)
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]
    return f"Predicted Wine Quality (0-10): {pred:.2f}"

app = gr.Interface(
    fn=predict_wine_quality,
    inputs=[gr.Number(label=f) for f in feature_names],
    outputs=gr.Text(label="Wine Quality Prediction"),
    title="Wine Quality Prediction (ANN)",
    description="Enter RAW physicochemical properties from the dataset to get wine quality prediction."
)

if __name__ == "__main__":
    app.launch()

import os
print("Model file path loaded:", os.path.abspath("wine_ann_model.joblib"))
print("Scaler file path loaded:", os.path.abspath("wine_scaler.joblib"))
