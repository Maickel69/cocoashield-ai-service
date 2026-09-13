import requests
import io
from PIL import Image

def test_health():
    print("Testing /health endpoint...")
    try:
        r = requests.get("http://127.0.0.1:8000/health")
        print("Health status code:", r.status_code)
        print("Health response:", r.json())
        return r.status_code == 200
    except Exception as e:
        print("Health check failed:", e)
        return False

def test_predict():
    print("\nTesting /predict endpoint with test image...")
    img = Image.new("RGB", (224, 224), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    try:
        files = {"file": ("test_cacao.jpg", buf, "image/jpeg")}
        r = requests.post("http://127.0.0.1:8000/predict", files=files)
        print("Predict status code:", r.status_code)
        print("Predict response:", r.json())
        return r.status_code == 200
    except Exception as e:
        print("Predict test failed:", e)
        return False

if __name__ == "__main__":
    test_health()
    test_predict()
