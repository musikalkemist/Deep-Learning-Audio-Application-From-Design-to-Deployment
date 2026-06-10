import requests

URL = "http://127.0.0.1/predict"
TEST_AUDIO_FILE_PATH = "test/left.wav"

if __name__ == "__main__":

    audio_file = open(TEST_AUDIO_FILE_PATH, "rb")
    values = {"file": (TEST_AUDIO_FILE_PATH, audio_file, "audio/wav")}
    response = requests.post(URL, files=values)

    if response.status_code == 200:
        data = response.json()
        print(f"Predicted keyword is: {data['keyword']}")
    else:
        print(f"Error: Request failed with status code {response.status_code}")
        print("Server answer:", response.text)