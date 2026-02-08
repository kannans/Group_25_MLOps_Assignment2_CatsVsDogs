import requests
import time

def test_health_endpoint():
    url = "http://localhost:8000/health"
    for _ in range(5):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("Health check passed!")
                return True
        except requests.exceptions.ConnectionError:
            print("Waiting for service to start...")
            time.sleep(2)
    return False

if __name__ == "__main__":
    if test_health_endpoint():
        print("Smoke tests passed successfully.")
    else:
        print("Smoke tests failed.")
        exit(1)
