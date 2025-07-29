import requests


def test_get_root():
    response = requests.get("http://127.0.0.1:8000/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]


def test_post_inference():
    data = {
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
        "native-country": "United-States"
    }
    response = requests.post("http://127.0.0.1:8000/data/", json=data)
    assert response.status_code == 200
    assert "result" in response.json()
