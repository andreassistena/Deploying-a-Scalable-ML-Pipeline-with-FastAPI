# local_api.py
import json
import requests

BASE_URL = "http://127.0.0.1:8000"

def main():
    # ---- GET /
    r_get = requests.get(f"{BASE_URL}/")
    print("GET / -> Status:", r_get.status_code)
    try:
        print("GET / -> Body:", r_get.json())
    except Exception:
        print("GET / -> Raw:", r_get.text)

    # ---- POST /data/
    # Note: keys must match the schema exactly (hyphenated names included)
    payload = {
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

    headers = {"Content-Type": "application/json"}
    r_post = requests.post(f"{BASE_URL}/data/", headers=headers, data=json.dumps(payload))
    print(f"POST /data/ -> Status: {r_post.status_code}")
    try:
        print("POST /data/ -> Body:", r_post.json())
    except Exception:
        print("POST /data/ -> Raw:", r_post.text)

if __name__ == "__main__":
    main()