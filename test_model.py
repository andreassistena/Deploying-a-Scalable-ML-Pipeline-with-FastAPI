import os
import numpy as np
import pandas as pd
from ml.model import load_model, inference, compute_model_metrics
from ml.data import process_data

# Load artifacts
model = load_model(os.path.join("model", "model.pkl"))
encoder = load_model(os.path.join("model", "encoder.pkl"))
lb = load_model(os.path.join("model", "lb.pkl"))

# Sample data
sample_data = pd.DataFrame({
    "age": [37],
    "workclass": ["Private"],
    "fnlgt": [178356],
    "education": ["HS-grad"],
    "education-num": [10],
    "marital-status": ["Married-civ-spouse"],
    "occupation": ["Prof-specialty"],
    "relationship": ["Husband"],
    "race": ["White"],
    "sex": ["Male"],
    "capital-gain": [0],
    "capital-loss": [0],
    "hours-per-week": [40],
    "native-country": ["United-States"],
})

cat_features = [
    "workclass", "education", "marital-status",
    "occupation", "relationship", "race", "sex", "native-country"
]

X, y, _, _ = process_data(
    sample_data, categorical_features=cat_features,
    label=None, training=False, encoder=encoder, lb=lb
)


# Test 1: Model type
def test_model_type():
    from sklearn.ensemble import RandomForestClassifier
    assert isinstance(model, RandomForestClassifier)


# Test 2: Prediction shape
def test_prediction_shape():
    preds = inference(model, X)
    assert preds.shape == (1,), "Prediction output shape is incorrect"

# Test 3: Evaluation metrics thresholds
def test_model_metrics():
    # Simulated known values for test
    y_true = np.array([1])
    y_pred = np.array([1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert precision >= 0.5, "Precision below threshold"
    assert recall >= 0.5, "Recall below threshold"
    assert fbeta >= 0.5, "F1 score below threshold"
