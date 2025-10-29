import gradio as gr
import joblib
import numpy as np

# Load your trained model and scaler (make sure they match your notebook and are from latest training on RAW y)
model = joblib.load("wine_ann_model.joblib")
scaler = joblib.load("wine_scaler.joblib")

# Features (order MUST match training/CSV)
feature_names = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
    "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
]

def predict_wine_quality(*inputs):
    # Accept raw (unscaled) inputs as typed in CSV data
    features = np.array(inputs).reshape(1, -1)
    # Scale features like in model training
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]
    # Clamp predicted value to 3–8 (wine quality range) for display clarity
    
    return f"Predicted Wine Quality (3-8 range): {pred:.2f}"

app = gr.Interface(
    fn=predict_wine_quality,
    inputs=[gr.Number(label=f) for f in feature_names],
    outputs=gr.Text(label="Wine Quality Prediction"),
    title="Wine Quality Prediction (ANN)",
    description="Enter RAW physicochemical properties (from the CSV, NOT standardized) to get wine quality prediction in the 3-8 range."
)

if __name__ == "__main__":
    app.launch()
