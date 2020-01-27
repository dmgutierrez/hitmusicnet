# -*- coding: utf-8 -*-
from helper.data_manager import DataManager
from helper.custom_log import init_logger
from keras.models import model_from_json
import tensorflow as tf
import os
import pandas as pd

# Init Global Parameters
config_file = None
host = None
port = None
tracks_index = None
artist_index = None
album_index = None
audio_feat_index = None
text_feat_index = None
audio_dir = None
doc_type = None
indexNames = None
countries = None
data_manager = None
thread = None
model_args = None
logger = None
graph = None
musicPopNet_cls = None
x_test = None

os.environ["Audio_dir"] = "C:\\Users\\dmarg\\Desktop\\David\\Projects\\master-thesis-music-analysis\\audioFiles"
#"E:\\David\\Datasets\\TFM\\audioFiles"

def load_model():

    global musicPopNet_cls
    try:
        with open(os.path.join('saved_models', 'class_feature_compressed_auto', 'model_1_A', 'model.json'), 'r') as f:
            musicPopNet_cls = model_from_json(f.read())

        musicPopNet_cls.load_weights(os.path.join('saved_models',
                                                  'class_feature_compressed_auto',
                                                  'model_1_A', 'weights.h5'))
    except Exception as e:
        msg = str(
            e) + '. WARNING: The models are not computed yet. After the training process the models will be available'
        logger.warning(msg)
        musicPopNet_cls = None


def init():
    global config_file
    global host
    global port
    global audio_dir
    global doc_type
    global indexNames
    global countries
    global data_manager
    global thread
    global model_args
    global logger
    global graph
    global musicPopNet_cls
    global x_test
    global tracks_index
    global artist_index
    global album_index
    global audio_feat_index
    global text_feat_index

    thread = None
    config_file = 'config.ini'
    host= '127.0.0.1'
    port = '5001'
    doc_type = ['doc', 'doc', 'doc']
    tracks_index = 'spotify_tracks'
    artist_index = 'spotify_artists'
    album_index = 'spotify_albums'
    audio_feat_index = None
    text_feat_index = None


    logger = init_logger(__name__, testing_mode=False)

    x_test = pd.read_csv(os.path.join('input', 'x_test_regression.csv'), index_col=0)

    indexNames = ['spotify_tracks', 'spotify_artists', 'spotify_albums']
    countries  = ['BE', 'FI', 'AR', 'CR', 'AT', 'ES', 'FR', 'IT', 'GB',
                  'IE', 'NO', 'DK','BR','CL', 'MX', 'CO','SE', 'DE', 
                  'NL', 'US', 'AU', 'CA', 'ZA', 'IL','EC', 'PT']

    data_manager = DataManager(config_file, host=host, port=port, countries=countries)
    graph = tf.get_default_graph()