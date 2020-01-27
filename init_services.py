# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 17:20:40 2019

@author: dmarg
"""

from helper.data_helper import str2bool
from flask import Flask, request
from services import services as serv
from helper.data_helper import build_output
from helper import global_variables as gv
from threading import Thread
import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)


@app.route('/create_db/start', methods=['GET'])
def create_db_es_service():
    service = 'Music Popularity Prediction'
    task = 'Data Generation'
    status = 'Creating SpotGenTrack in Elasticsearch server ...'
    job = build_output(name=service, task=task, status=status)

    def batch_db_creation():        
        output=serv.create_db_elasticsearch()
        if output:
            status = 'The Dataset was created with Success! Check-out your ES indexes!'
        else:
            status = 'An error occurred during the process. Please try again!'
        output = build_output(name=service, task=task, status=status)
        return json.dumps(output)
    if gv.thread is None:
        thread = Thread(target=batch_db_creation)
        thread.start()
    return json.dumps(job)


@app.route('/audio_features/start', methods=['GET'])
def audio_low_level_feature_extraction_service():
    service = 'Music Popularity Prediction'
    task = 'Audio Features Extractor'
    status = 'Calculating Low-level Audio Features ...'
    job = build_output(name=service, task=task, status=status)

    def batch_audio_features():
        output = serv.collect_low_level_audio_features()
        if output:
            status = 'The audio low-level features were calculated with Success! Check-out your ES indexes!'
        else:
            status = 'An error occurred during the process. Checkout the logs. Please try again!'
        output = build_output(name=service, task=task, status=status)
        return json.dumps(output)
    if gv.thread is None:
        thread = Thread(target=batch_audio_features)
        thread.start()
    return json.dumps(job)


@app.route('/text_features/start', methods=['GET', 'POST'])
def text_feature_extraction_service():
    job = 'Calculating Text Features ...'

    def batch_text_features():
        output = serv.collect_text_features()
        if output:
            output = 'The set of text features were calculated with Success! Check-out your ES indexes!'
        else:
            output = 'An error occurred during the process. Please try again!'
        return output
    if gv.thread is None:
        thread = Thread(target=batch_text_features)
        thread.start()
    return job


@app.route('/train_popularity/start', methods=['GET', 'POST'])
def training_popularity_deep_model():
    model = str(request.headers.get('model'))
    prep = str2bool(str(request.headers.get('preprocessing')))
    new_data = str2bool(str(request.headers.get('new_data')))
    compress = str(request.headers.get('compress'))
    task = str(request.headers.get('task'))
    output = serv.train_popnet(new_data=new_data, model=model, new_preprocessing=prep, compress=compress,
                               task=task)
    return json.dumps(output)

@app.route('/predict_popularity/start/<track_id>', methods=['GET'])
def predict_popularity(track_id):
    output = serv.predict_musicPopNet(track_id=track_id)
    return json.dumps(output)


if __name__ == '__main__':
    gv.init()
    app.run(host="192.168.0.22", port="5000", debug=False)