import requests
import time
from PIL import Image
import io


def test_endpoints():
    url_health = "http://localhost:8000/health"
    url_predict = "http://localhost:8000/predict"

    # 1. Test Health Endpoint
    service_up = False
    for _ in range(5):
        try:
            response = requests.get(url_health)
            if response.status_code == 200:
                print("Health check passed!")
                service_up = True
                break
        except requests.exceptions.ConnectionError:
            print("Waiting for service to start...")
            time.sleep(2)

    if not service_up:
        print("Service did not start correctly.")
        return False

    # 2. Test Predict Endpoint
    try:
        # Create a dummy image
        img = Image.new("RGB", (224, 224), color="red")
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)

        files = {"file": ("dummy.jpg", img_byte_arr, "image/jpeg")}
        predict_response = requests.post(url_predict, files=files)

        if predict_response.status_code == 200:
            result = predict_response.json()
            if "prediction" in result:
                print(f"Prediction test passed! Result: {result['prediction']}")
                return True
            else:
                print(
                    f"Prediction test failed, missing prediction in response: {result}"
                )
        else:
            print(f"Prediction endpoint returned status {predict_response.status_code}")
    except Exception as e:
        print(f"Prediction test failed with exception: {e}")

    return False


if __name__ == "__main__":
    if test_endpoints():
        print("Smoke tests passed successfully.")
    else:
        print("Smoke tests failed.")
        exit(1)
