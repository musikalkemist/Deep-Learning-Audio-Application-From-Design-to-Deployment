import argparse
import requests

URL = 'http://127.0.0.1/predict'
TEST_AUDIO_FILE_PATH = 'test/left.wav'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, required=False, default=TEST_AUDIO_FILE_PATH,
                        help='Enter the file path of the audio file.')
    parser.add_argument('--model_type', type=str, required=False, default='tensorflow',
                        help='Choose whether you want to run it with TensorFlow or PyTorch model (default: tensorflow). Supported values: tensorflow, pytorch')
    args = parser.parse_args()

    with open(args.filepath, 'rb') as f:
        file_values = {'file': (args.filepath, f, 'audio/wav')}
        data = {'model_type': args.model_type}
        response = requests.post(URL, data=data, files=file_values)
    data = response.json()
    print(f"Predicted keyword is: {data['keyword']}")
    
