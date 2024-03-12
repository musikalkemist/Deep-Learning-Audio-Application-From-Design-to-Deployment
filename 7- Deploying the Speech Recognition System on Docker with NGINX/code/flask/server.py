import os
import random

from flask import Flask, request, jsonify
from keyword_spotting_service import Keyword_Spotting_Service
from keyword_spotting_service_pt import Keyword_Spotting_Service as Keyword_Spotting_Service_pt

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    r"""Endpoint to predict keyword.

    Returns:
        json file with the following format:
        {
            'keyword': KEYWORD_VALUE
        }
    """
    # get file from POST request and save it
    audio_file = request.files['file']
    filename = str(random.randint(0, 100000))
    audio_file.save(filename)

    # instantiate keyword spotting service singleton and get prediction
    if request.form['model_type'] == 'tensorflow':
        kss = Keyword_Spotting_Service()
    elif request.form['model_type'] == 'pytorch':
        kss = Keyword_Spotting_Service_pt()
    else:
        raise ValueError('The only two supported model types are: tensorflow, pytorch')
    predicted_keyword = kss.predict(filename)

    # we don't need the audio file anymore - let's delete it!
    os.remove(filename)

    # send back result as a json file
    result = {'keyword': predicted_keyword}

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=False)
