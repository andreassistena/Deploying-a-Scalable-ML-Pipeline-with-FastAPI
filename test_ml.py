import pytest
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
    assert response.json() == {"message": "Welcome to the FastAPI ML Inference API!"}

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