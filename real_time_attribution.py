import flask
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import time
import shap
import os

app = Flask(__name__)

treated_model = None
control_model = None
feature_cols = None
scaler = None
treated_explainer = None
control_explainer = None

def load_models():
    global treated_model, control_model, feature_cols, scaler, treated_explainer, control_explainer
    
    model_dirs = ['models']
    
    for model_dir in model_dirs:
        try:
            treated_model = joblib.load(f'{model_dir}/treated_model.pkl')
            control_model = joblib.load(f'{model_dir}/control_model.pkl')
            feature_cols = joblib.load(f'{model_dir}/feature_cols.pkl')
            scaler = joblib.load(f'{model_dir}/scaler.pkl')
            
            treated_explainer = shap.TreeExplainer(treated_model)
            control_explainer = shap.TreeExplainer(control_model)
            
            print(f"Models loaded successfully from {model_dir}/")
            return True
        except Exception as e:
            print(f"Error loading models from {model_dir}/: {e}")
    
    print("Failed to load models from any directory")
    return False

@app.route('/health', methods=['GET'])
def health_check():
    if treated_model is None or control_model is None:
        return jsonify({"status": "error", "message": "Models not loaded"}), 500
    return jsonify({"status": "healthy", "message": "API is running and models are loaded"})

@app.route('/predict_uplift', methods=['POST'])
def predict_uplift():
    start_time = time.time()
    
    if treated_model is None or control_model is None:
        if not load_models():
            return jsonify({"error": "Models not loaded. Please try again later."}), 500
    
    data = request.json
    
    if not data:
        return jsonify({"error": "No input data provided"}), 400
    
    input_df = pd.DataFrame([data])
    
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_data = preprocess_single_observation(input_df, feature_cols, scaler)
    
    p_treated = treated_model.predict_proba(input_data[feature_cols])[0, 1]
    p_control = control_model.predict_proba(input_data[feature_cols])[0, 1]
    uplift = p_treated - p_control
    
    feature_attributions = calculate_feature_attributions(input_data)
    
    process_time = time.time() - start_time
    
    recommendation = "TREAT" if uplift > 0.01 else "DO NOT TREAT" if uplift <= 0 else "CONSIDER"
    
    return jsonify({
        'uplift': float(uplift),
        'conversion_probability': float(p_treated),
        'baseline_probability': float(p_control),
        'recommendation': recommendation,
        'feature_attributions': feature_attributions,
        'process_time_ms': process_time * 1000
    })

def preprocess_single_observation(df, feature_cols, scaler):
    base_features = [col for col in feature_cols if '_interaction' not in col]
    
    for i, f1 in enumerate(base_features):
        for f2 in base_features[i+1:]:
            interaction_name = f"{f1}_{f2}_interaction"
            if interaction_name in feature_cols:
                df[interaction_name] = df[f1] * df[f2]
    
    df[feature_cols] = scaler.transform(df[feature_cols])
    
    return df

def calculate_feature_attributions(input_data):
    global treated_explainer, control_explainer
    
    treated_shap = treated_explainer.shap_values(input_data[feature_cols])
    control_shap = control_explainer.shap_values(input_data[feature_cols])
    
    if isinstance(treated_shap, list):
        treated_shap = treated_shap[1]
        control_shap = control_shap[1]
    
    uplift_shap = treated_shap - control_shap
    
    return {feature: float(uplift_shap[0][i]) for i, feature in enumerate(feature_cols)}

@app.route('/predict_uplift_batch', methods=['POST'])
def predict_uplift_batch():
    start_time = time.time()
    
    if treated_model is None or control_model is None:
        if not load_models():
            return jsonify({"error": "Models not loaded. Please try again later."}), 500
    
    data = request.json
    
    if not data or not isinstance(data, list):
        return jsonify({"error": "Input must be a list of observations"}), 400
    
    input_df = pd.DataFrame(data)
    
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_data = preprocess_batch_observations(input_df, feature_cols, scaler)
    
    p_treated = treated_model.predict_proba(input_data[feature_cols])[:, 1]
    p_control = control_model.predict_proba(input_data[feature_cols])[:, 1]
    uplift = p_treated - p_control
    
    process_time = time.time() - start_time
    
    results = []
    for i in range(len(data)):
        recommendation = "TREAT" if uplift[i] > 0.01 else "DO NOT TREAT" if uplift[i] <= 0 else "CONSIDER"
        results.append({
            'uplift': float(uplift[i]),
            'conversion_probability': float(p_treated[i]),
            'baseline_probability': float(p_control[i]),
            'recommendation': recommendation
        })
    
    return jsonify({
        'results': results,
        'process_time_ms': process_time * 1000
    })

def preprocess_batch_observations(df, feature_cols, scaler):
    base_features = [col for col in feature_cols if '_interaction' not in col]
    
    for i, f1 in enumerate(base_features):
        for f2 in base_features[i+1:]:
            interaction_name = f"{f1}_{f2}_interaction"
            if interaction_name in feature_cols:
                df[interaction_name] = df[f1] * df[f2]
    
    df[feature_cols] = scaler.transform(df[feature_cols])
    
    return df

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "name": "Uplift Modeling API",
        "description": "API for real-time conversion uplift predictions",
        "version": "1.0",
        "endpoints": {
            "/": "This documentation",
            "/health": "Health check endpoint (GET)",
            "/predict_uplift": "Single prediction endpoint (POST)",
            "/predict_uplift_batch": "Batch prediction endpoint (POST)"
        },
        "usage": {
            "single_prediction": {
                "method": "POST",
                "endpoint": "/predict_uplift",
                "content_type": "application/json",
                "example_body": {
                    "f0": 0.5, "f1": -1.2, "f2": 0.3, "f3": 0.7, 
                    "f4": -0.2, "f5": 0.1, "f6": 0.8, "f7": -0.5, 
                    "f8": 0.4, "f9": 0.2, "f10": -0.3, "f11": 0.6
                }
            },
            "batch_prediction": {
                "method": "POST",
                "endpoint": "/predict_uplift_batch",
                "content_type": "application/json",
                "example_body": [
                    {"f0": 0.5, "f1": -1.2, "f2": 0.3},
                    {"f0": 0.3, "f1": 0.7, "f2": -0.2}
                ]
            }
        }
    })

# Add this at the end of your file
if __name__ == '__main__':
    # Load models when the app starts
    with app.app_context():
        load_models()
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 8080))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=True)