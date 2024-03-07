import librosa
import os
import json

DATASET_PATH = "dataset"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050 # 1 sec. of audio


def preprocess_dataset(dataset_path: str, json_path: str, num_mfcc: int = 13, n_fft: int = 2048, hop_length: int = 512):
    r"""Extract MFCCs from audio file.

    Args:
        dataset_path: `str`
            Path to dataset.
        json_path: `str`
            Path to json file used to save MFCCs.
        num_mfcc: `int`, default=13
            Number of MFCC coefficients.
        n_fft: `int`, default=2048
            Length of the FFT window. Measured in # of samples.
        hop_length: `int`, default=512
            Sliding window for FFT. Number of samples between successive frames.

    Returns:
    """
    # dictionary where we'll store mapping, labels, MFCCs and filenames
    data = {
        'mapping': [],  # mappings of label names to label ids
        'labels': [],   # y
        'MFCCs': [],    # x
        'files': []     # original files for each (x, y) pair
    }

    # loop through all the subdirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        filenames = [f for f in filenames if f.endswith('wav')]
        # ensure we are at subfolder level
        if dirpath is not dataset_path:
            label_name = dirpath.rsplit('/')[-1]
            data['mapping'].append(label_name)
            print(f'\nProcessing: \"{label_name}\"')

            # loop through all the filenames and extract MFCCs
            for f in filenames:
                filepath = os.path.join(dirpath, f)

                # load audio file and slice it to ensure length consistency among different files
                signal, sample_rate = librosa.load(filepath)

                # drop audio files with less than pre-decided number of samples
                if len(signal) >= NUM_SAMPLES_TO_CONSIDER:
                    signal = signal[:NUM_SAMPLES_TO_CONSIDER]  # enforce 1 sec long signals
                    MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    data['labels'].append(i-1)
                    data['MFCCs'].append(MFCCs.T.tolist())
                    data['files'].append(filepath)
                    print(f'{filepath}: {i - 1}')

    # save data in json file
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__:
    preprocess_dataset(DATASET_PATH, JSON_PATH)
