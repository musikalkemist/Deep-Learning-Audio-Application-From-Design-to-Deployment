import argparse
import requests

# server url
URL = "http://127.0.0.1:5050/predict"


# audio file we'd like to send for predicting keyword
FILE_PATH = "test/left.wav"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, required=False, default=FILE_PATH,
                        help='Enter the file path of the audio file.')
    args = parser.parse_args()
    
    # open files
    file = open(args.filepath, "rb")

    # package stuff to send and perform POST request
    values = {"file": (args.filepath, file, "audio/wav")}
    response = requests.post(URL, files=values)
    data = response.json()

    print("Predicted keyword: {}".format(data["keyword"]))












