# -*- coding: utf-8 -*-
import configparser as cp
import os
from hdfs import InsecureClient
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
import itertools
import datetime
import socket


def ConfigSectionMap(config, section):
    dict_data = {}
    options = config.options(section)
    for option in options:
        try:
            dict_data[option] = config.get(section, option)
            if dict_data[option] == -1:
                print("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict_data[option] = None
    return dict_data


def read_credentials(filename='config.ini', platform= 'SPOTIFY'):
    config = cp.ConfigParser()
    config.read(filename)
    if platform=='SPOTIFY':
        cid = ConfigSectionMap(config, 'SPOTIFY CLIENT CREDENTIALS')['client_id']
        secret = ConfigSectionMap(config, 'SPOTIFY CLIENT CREDENTIALS')['client_secret']
        username = ConfigSectionMap(config, 'SPOTIFY CLIENT CREDENTIALS')['username']
        cred = (cid, secret, username)
    else:
        cid = ConfigSectionMap(config, 'GENIUS CLIENT CREDENTIALS')['client_id']
        secret = ConfigSectionMap(config, 'GENIUS CLIENT CREDENTIALS')['client_secret']
        access_token = ConfigSectionMap(config, 'GENIUS CLIENT CREDENTIALS')['access_token']
        cred = (cid, secret, access_token)
    return cred

def read_data_sources(filename='config.ini'):
    config = cp.ConfigParser()
    config.read(filename)
    sources = {}
    
    file_audio_HL = ConfigSectionMap(config, 'DATA SOURCES')['filename_audio_hl']
    file_audio_LL = ConfigSectionMap(config, 'DATA SOURCES')['filename_audio_ll']
    file_lyrics = ConfigSectionMap(config, 'DATA SOURCES')['filename_lyrics']
    file_semantic = ConfigSectionMap(config, 'DATA SOURCES')['filename_semantic']
    file_prev_audio = ConfigSectionMap(config, 'DATA SOURCES')['filename_prev']
    file_pop_pred = ConfigSectionMap(config, 'DATA SOURCES')['filename_pop_pred']
    file_best_weights = ConfigSectionMap(config, 'DATA SOURCES')['filename_best_weights']
    file_music_dataset = ConfigSectionMap(config, 'DATA SOURCES')['filename_music_dataset']
    file_final_res = ConfigSectionMap(config, 'DATA SOURCES')['filename_final_results']
    
    sources = {'Filename_audio_HL': file_audio_HL,
               'Filename_audio_LL': file_audio_LL,
               'Filename_lyrics':file_lyrics,
               'Filename_semantic':file_semantic,
               'Filename_prew_audio': file_prev_audio,
               'Filename_pop_pred': file_pop_pred,
               'Filename_best_weights':file_best_weights,
               'Filename_music_dataset':file_music_dataset,
               'Filename_final_results':file_final_res}
    
    return sources

def read_data_directories(filename='config.ini'):
    config = cp.ConfigParser()
    config.read(filename)
    directories = {}
    
    saved_dir = ConfigSectionMap(config, 'DATA DIRECTORIES')['saved_dir']
    data_dir = ConfigSectionMap(config, 'DATA DIRECTORIES')['data_dir']
    saved_models = ConfigSectionMap(config, 'DATA DIRECTORIES')['saved_models']
    saved_audio_dir = ConfigSectionMap(config, 'DATA DIRECTORIES')['saved_audio_dir']
    cv_dir = ConfigSectionMap(config, 'DATA DIRECTORIES')['cv_dir']
    ml_dataset_dir = ConfigSectionMap(config, 'DATA DIRECTORIES')['datasets_dir']
    final_res_dir = ConfigSectionMap(config, 'DATA DIRECTORIES')['evaluation_dir']
    
    directories = {'saved_directory': saved_dir,
                   'data_directory': data_dir,
                   'saved_audio_directory':saved_audio_dir,
                   'saved_models':saved_models,
                   'cv_dir':cv_dir,
                   'ml_dataset_dir':ml_dataset_dir,
                   'validation_res_dir': final_res_dir}
    return directories


def get_date():
     current_date = datetime.datetime.now()
     final_date = current_date.strftime("%Y%m%d")
     return final_date
 
    
def save_dataframe(filename, df, root_dir='data', enc='utf8'):
    data_dir = os.path.join(root_dir, filename)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    df.to_csv(data_dir,encoding=enc)
    print('Data Stored correctly!\n')
    return True

def read_dataframe(filename, root_dir = 'data', enc='utf8'):
    data_dir = os.path.join(root_dir, filename)
    
    tp  = pd.read_csv(data_dir, encoding=enc, index_col=0,
                      iterator=True, chunksize=100)
    df = pd.concat(tp, ignore_index=True)
    
    df.apply(lambda x: pd.api.types.infer_dtype)

    return df
    
def save_dataframe_as_hdfs(filename, df, root_dir='data',enc='utf8'):
    data_dir = os.path.join(root_dir, filename)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    client_hdfs = InsecureClient('http://' + os.environ['IP_HDFS'] + ':50070')
    
    with client_hdfs.write(data_dir, encoding = enc) as writer:
        df.to_csv(writer)
    return True

def read_hdfs(filename, root_dir='data'):
    data_dir = os.path.join(root_dir, filename)
    client_hdfs = InsecureClient('http://' + os.environ['IP_HDFS'] + ':50070')
    
    with client_hdfs.read(data_dir, encoding = 'latin-1') as reader:
        df = pd.read_csv(reader,index_col=0)
    return df


def get_text_language(language):
    valid_language = ''
    
    if language =='en':
        valid_language = 'english'
    elif language == 'ar':
        valid_language = 'arabic'
    elif language == 'az':
        valid_language = 'azerbaijani'
    elif language == 'da':
        valid_language = 'danish'
    elif language == 'nl':
        valid_language = 'dutch'
    elif language == 'fr':
        valid_language = 'french'
    elif language == 'de':
        valid_language = 'german'
    elif language == 'el':
        valid_language = 'greek'
    elif language == 'hu':
        valid_language = 'hungarian'
    elif language == 'id':
        valid_language = 'indonesian'
    elif language == 'it':
        valid_language = 'italian'
    elif language == 'kk':
        valid_language = 'kazakh'
        #
    elif language == 'ne':
        valid_language = 'nepali'
    elif language == 'no':
        valid_language = 'norwegian'
    elif language == 'pt':
        valid_language = 'portuguese'
    elif language == 'ro':
        valid_language = 'romanian'
    elif language == 'ru':
        valid_language = 'russian'
    #
    elif language == 'es':
        valid_language = 'spanish'
    elif language == 'sv':
        valid_language = 'swedish'
    elif language == 'tr':
        valid_language = 'turkish'
    else:
        valid_language = 'NOT VALID' # By default
    
    return valid_language
 
def do_analysis(row):
    error_message = 'we are not licensed to display the full lyrics for this song at the moment'    
    if (row is not np.nan and row is not '' and
        row is not None and error_message not in row):
        return True
    else:
        return False
    
 
def clean_song_name(song_name):
    # Remove parenthesis
    done = 0
    while done !=1:
        start = song_name.find( '(' )
        end = song_name.find( ')' )
        if start != -1 and end != -1:
            result = song_name[start+1:end]
            removed_str = '(' + result + ')'
            song_name = song_name.replace(removed_str, '')
        else:
            done = 1
            
    # Remove brakets
    done = 0
    while done !=1:
        start = song_name.find( '[' )
        end = song_name.find( ']' )
        if start != -1 and end != -1:
            result = song_name[start+1:end]
            removed_str = '[' + result + ']'
            song_name = song_name.replace(removed_str, '')
        else:
            done = 1
    
    # Remove remix
    song_name = song_name.replace('remix', '').replace('Remix', '')
    
    # Remove dash
    start = song_name.find(' - ')
    if start != -1:
        removed_str = song_name[start:]
        song_name = song_name.replace(removed_str, '')
    return song_name

def check_infinity_data(history):
    history_new = history
    # Check if there is an infinity data point
    if False in np.isfinite(history):
        # Replace nans with 99
        valid_idx = list(np.where(np.isfinite(history))[0])
        for idx, val in enumerate(history):
            # Not finite value
            if idx not in valid_idx:
                history_new[idx] = 99
    return history_new


def get_model_subdir(semantic_an=True, metadata=True, audio_LL=True):
    model_subDir = ''
    if not semantic_an and not metadata and not audio_LL:
        print('Not valid Analysis')
        return None
    elif not semantic_an and not metadata and audio_LL:
        model_subDir = 'audio_models'
    
    elif not semantic_an and metadata and not audio_LL:
        model_subDir = 'metadata_models'  
    
    elif not semantic_an and metadata and audio_LL:
        model_subDir = 'metadata_audio_models'    
    
    elif semantic_an and not metadata and not audio_LL:
        model_subDir = 'semantic_models'
    
    elif semantic_an and not metadata and audio_LL:
        model_subDir = 'semantic_audio_models'
        
    elif semantic_an and metadata and not audio_LL:
        model_subDir = 'semantic_metadata_models'
        
    elif semantic_an and metadata and audio_LL:
        model_subDir = 'semantic_metadata_audio_models'
    
    return model_subDir


def get_popular_genres():    
    WIKI_URL = "https://en.wikipedia.org/wiki/List_of_popular_music_genres"
    req = requests.get(WIKI_URL)
    b = BeautifulSoup(req.content, 'lxml')
    req = requests.get(WIKI_URL)
    b = BeautifulSoup(req.content, 'html.parser')
    links = []
    # in this case, all of the links we're in a '<li>' brackets.
    for i in b.find_all(name = 'li'):
        links.append(i.text)
    
    general_genres = {'African':links[81:127], 'Asian':links[128:132],
                      'East Asian':links[133:145],
                      'South & southeast Asian':links[146:164],
                      'Avant-garde':links[165:169],'Blues':links[170:196],
                      'Caribbean':links[197:233],  'Comedy':links[234:237],
                      'Country':links[238:273],    'Easy listening':links[274:280],
                      'Electronic':links[280:504], 'Folk':links[505:524],
                      'Hip hop & Rap':links[525:571],    'Jazz':links[572:623],
                      'Latin':links[624:687],      'Pop':links[688:755],
                      'R&B & Soul':links[756:774],   'Rock':links[775:919]}
    
    for key, list_genre in general_genres.items():
        clean_genres = []
        clean_genres = [g.split('\n') for g in list_genre]
        all_clean_genres = list(itertools.chain.from_iterable(clean_genres))
        
        # Remove duplicate values
        set_genres = list(set(all_clean_genres))
        general_genres[key] = [g.lower().replace('-', ' ') for g in set_genres]
    # Add edm to Electronic
    general_genres['Electronic'].append('edm')
    general_genres['Electronic'].append('house')
    general_genres['African'].append('afrobeats')
    general_genres['African'].append('afropop')
    general_genres['Latin'].remove('folk')
    
    return general_genres

def available_data_spotify_features():
    spotify_features = {'acousticness':True,'danceability':True,'energy':True,'duration_ms':True,
                        'instrumentalness':True,'liveness':True,'loudness':True, 'mode':True, 'key':True,
                        'speechiness':True,'tempo':True, 'valence':True, 'Name':True, 'Artist':True,
                        'Artist_Popularity':True, 'Artist_followers':True, 
                        'Genre':False, 'Release Date':False,
                        'Playlist':True, 'id':True}

    return spotify_features

def available_data_popularity():
    popularity_data = {'Popularity':True, 'Popularity_Class':True,'General_genre':True}
    return popularity_data

def available_data_audio_features(mfcc=(40,True),chromagram=(12,True),melSpectrogram=(128,True), spect_contr=(7,True),
                                  tonnetz=(6,True)):
    cols_mfccs = ['MFCSS_' + str(i + 1) for i in range(mfcc[0])]
    k = [mfcc[1] for i in range(mfcc[0])]
    audio_features = dict(zip(cols_mfccs, k))

    cols_chroma = ['Chroma_' + str(i + 1) for i in range(chromagram[0])]
    k = [chromagram[1] for i in range(chromagram[0])]
    audio_features .update(zip(cols_chroma, k))

    cols_mel = ['Mel_' + str(i + 1) for i in range(melSpectrogram[0])]
    k = [melSpectrogram[1] for i in range(melSpectrogram[0])]
    audio_features.update(zip(cols_mel, k))

    cols_contrast = ['Spectral_contrast_' + str(i + 1) for i                 in range(spect_contr[0])]
    k = [spect_contr[1] for i in range(spect_contr[0])]
    audio_features.update(zip(cols_contrast, k))

    cols_tonnetz = ['Tonnetz_' + str(i + 1) for i in range(tonnetz[0])]
    k = [tonnetz[1] for i in range(tonnetz[0])]
    audio_features.update(zip(cols_tonnetz, k))

    return audio_features


def available_data_semantic_features():
    semantic_features = {'Flesch_reading_ease':True, 'Sentence_similarity':True, 'Freq_distribution_coeff':True,
                         'lexical_coefficient':True, 'Average_syllables_length':True, 'Average_sentences_length':True,
                         'Number_sentebnes':True}
    return semantic_features


def load_X_y_data(data_dir=None, filenames=None):
    # Filenames is a list where the first position belongs to X_train file and the second to y_train
    X = read_dataframe(root_dir=data_dir, filename=filenames[0])
    y = read_dataframe(root_dir=data_dir, filename=filenames[1])
    return X, y

def get_open_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("",0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return str(port)

def str2bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError('Not Valid Input')

def build_output(name, task, status):
    output = {'Service': name, 'Task': task, 'Status': status}
    return output