import librosa
import numpy as np
import os
import torch

NUM_SAMPLES_TO_CONSIDER = 22050  # 1 sec. of audio
SAVED_MODEL_PATH = 'model.pt'

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class _Keyword_Spotting_Service:
    """Singleton class for keyword spotting inference with trained models."""
    model = None
    _mapping = [
        "down",
        "off",
        "on",
        "no",
        "yes",
        "stop",
        "up",
        "right",
        "left",
        "go"
    ]
    _instance = None

    @staticmethod
    def preprocess(filepath, num_mfcc: int = 13, n_fft: int = 2048, hop_length: int = 512):
        r"""Extract MFCCs from audio file.

        Args:
            filepath: `str`
                Path of audio file.
            num_mfcc: `int`, default=13
                Number of MFCC coefficients.
            n_fft: `int`, default=2048
                Length of the FFT window. Measured in # of samples.
            hop_length: `int`, default=512
                Sliding window for FFT. Number of samples between successive frames.

        Returns:
            MFCCs: `np.ndarray`
                2-dim array with MFCC data of shape (# time steps, # coefficients).
        """
        signal, sample_rate = librosa.load(filepath)  # load audio file
        if len(signal) >= NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]  # enforce 1 sec long signals
        # extract MFCCs
        MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        return MFCCs.T

    def predict(self, filepath: str):
        r"""Predict the keyword of the `filepath`.

        Args:
            filepath: `str`
                Path to audio file to predict.

        Returns:
            predicted_keyword: `np.ndarray`
                Keyword predicted by the model.
        """
        MFCCs = _Keyword_Spotting_Service.preprocess(filepath)  # extract MFCC
        MFCCs = MFCCs[np.newaxis, np.newaxis, ...]  # in PyTorch, model expects input in NCHW format (instead of NHWC format)
        MFCCs_pt = torch.from_numpy(MFCCs).to(dtype=torch.float32)
        prediction_logits = self.model(MFCCs_pt)
        predictions_pt = torch.argmax(prediction_logits, 1)
        predictions = np.array(self._mapping)[predictions_pt]
        return predictions


def Keyword_Spotting_Service():
    r"""Factory function for Keyword_Spotting_Service class.

        Returns:
        : _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service)

    """
    # ensure an instance is created only the first time the factory function is called
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        print('Loading the PyTorch model for only the first time.')
        _Keyword_Spotting_Service.model = torch.jit.load(SAVED_MODEL_PATH)
        _Keyword_Spotting_Service.model.eval()
    return _Keyword_Spotting_Service._instance


if __name__ == '__main__':
    # create 2 instances of the keyword spotting service
    kss = Keyword_Spotting_Service()
    kss1 = Keyword_Spotting_Service()

    # check that different instances of the keyword spotting service point back to the same object (singleton)
    assert kss is kss1

    # make a prediction
    keyword = kss.predict('test/down.wav')
    print(keyword)
