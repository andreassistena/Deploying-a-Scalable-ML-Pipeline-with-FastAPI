<<<<<<< HEAD
import pytest
# TODO: add necessary import
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from ml.model import train_model, compute_model_metrics

# Implement the first test. Change the function name and input as needed
def test_model_type():
    """
    # Test that train_model returns a RandomForestClassifier instance when trained on a simple dataset.
    """
    X = pd.DataFrame({
        "feature1": [0, 1] * 10,
        "feature2": [1, 0] * 10
    })
    y = pd.Series([0, 1] * 10)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier"


# Implement the second test. Change the function name and input as needed
def test_compute_model_metrics_values():
    """
    # Test that compute_model_metrics returns expected metric values (approximate).
    """
    y_true = np.array([0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)

    assert np.isclose(precision, 0.99, atol=1.0), "Precision out of expected range"
    assert np.isclose(recall, 0.66, atol=1.0), "Recall out of expected range"
    assert np.isclose(f1, 0.66, atol=1.0), "F1 out of expected range"
    


# Implement the third test. Change the function name and input as needed
def test_training_input_shape():
    """
    # Test that train_model handles correct input shape and type.
    """
    X = pd.DataFrame({
        "feature1": [0, 1] * 10,
        "feature2": [1, 0] * 10
    })
    y = pd.Series([0, 1] * 10)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y)
    model = train_model(X_train, y_train)
    assert hasattr(model, "predict"), "Trained model does not have predict method"
=======
from fastapi.testclient import TestClient
from main import app
from ml.model import load_model
import os

project_path = os.getcwd()
lb = load_model(os.path.join(project_path, "model", "lb.pkl"))
client = TestClient(app)


def test_one():
    """
    Test that the root endpoint returns the welcome message.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the FastAPI ML Inference API!"}


def test_two():
    """
    Test that the model makes a prediction on valid input data.
    """
    sample = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 178356,
        "education": "HS-grad",
        "education-num": 10,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United States"
    }
    response = client.post("/data/", json=sample)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] in ["<=50K", ">50K"]


def test_three():
    """
    Test that the API returns 422 when required fields are missing.
    """
    sample = {
        "age": 37,
        "education": "HS-grad"
    }
    response = client.post("/data/", json=sample)
    assert response.status_code == 422
>>>>>>> 7b99ec5a894c135ed8e7a3294919c18e6b8a9155
