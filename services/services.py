from helper import global_variables as gv
from models.model_selection import select_model
import pandas as pd
import numpy as np
import os


def create_db_elasticsearch():
    ok=False
    try:
        if gv.thread is None:
            dm = gv.data_manager
            ok=dm.generate_playlist_db()
    except Exception as e:
        gv.logger.error(e)
    return ok


def collect_low_level_audio_features(index_origin='spotify_tracks',
                                     doc_type_origin='doc',
                                     doc_type_featu='doc',
                                     index_features='low_level_audio_features',
                                     stat='mean'):
    ok=False
    try:
        gv.init()
        dm = gv.data_manager
        query = {"query": {"match_all": {}}}
        
        data_tracks = dm.extract_data_from_elasticsearch(index=index_origin,
                                                         doc_type=doc_type_origin,
                                                         query=query)
        # Remove Duplicates
        data_tracks.drop_duplicates(subset=['id'], keep='first', inplace=True)
        # Collect Low-level Features
        ok = dm.collect_low_level_features_ES(df=data_tracks, stat=stat,
                                              index_name=index_features,
                                              doc_type=doc_type_featu,
                                              index_col='track_id')
    except Exception as e:
        gv.logger.error(e)
    return ok



def collect_text_features(index_origin='spotify_tracks',
                          doc_type_origin='Track',
                          doc_type_featu='Features',
                          index_features='text_features'):
    ok=False
    try:
        gv.init()
        dm = gv.data_manager
        query = {"_source":['id', 'lyrics'], "query": {"match_all": {}}}
        
        # Extract Lyrics and put them into a DataFrame
        data_tracks = dm.extract_data_from_elasticsearch(index=index_origin,
                                                         doc_type=doc_type_origin,
                                                         query=query)
        
        # Remove Duplicates
        data_tracks.drop_duplicates(subset=['id'], keep='first', inplace=True)
        
        # Collect Low-level Features
        ok = dm.collect_text_features_ES(df=data_tracks,
                                         index_name=index_features,
                                         doc_type=doc_type_featu,
                                         index_col='track_id',
                                         max_cum=.80,
                                         mu=.40)
    except Exception as e:
        gv.logging.error(e)
    return ok


def train_popnet(new_data=False, model=1, new_preprocessing=False, compress='auto', task='regression'):
    try:
        gv.logger.info('Training PopNet for music popularity prediction.')
        dm = gv.data_manager
        query = {"query": {"match_all": {}}}
        if new_data:
            gv.logger.info('Extracting data from Elasticsearch ...')
            df_tracks = dm.extract_data_from_elasticsearch(index='spotify_tracks',
                                                           doc_type='Track',
                                                           query=query)
            gv.logger.info('Saving Data Locally')
            df_tracks.to_csv(os.path.join('input', 'df_tracks.csv'))
            
            df_artists = dm.extract_data_from_elasticsearch(index='spotify_artists',
                                                            doc_type='Artist',
                                                            query=query)
            gv.logger.info('Saving Data Locally')
            df_artists.to_csv(os.path.join('input', 'df_artists.csv'))

            df_audio_feat = dm.extract_data_from_elasticsearch(index='low_level_audio_features',
                                                               doc_type='Features',
                                                               query=query)
            gv.logger.info('Saving Data Locally')
            df_audio_feat.to_csv(os.path.join('input', 'df_audio_feat.csv'))

            df_text_feat = dm.extract_data_from_elasticsearch(index='text_features',
                                                              doc_type='Features',
                                                              query=query)
            gv.logger.info('Saving Data Locally')
            df_text_feat.to_csv(os.path.join('input', 'df_text_feat.csv'))
        
        else:
            gv.logger.info('Extracting data from CSV files ...')
            df_artists = pd.read_csv(os.path.join('input', 'df_artists.csv'), index_col=0)
            df_tracks = pd.read_csv(os.path.join('input', 'df_tracks.csv'), index_col=0)
            df_audio_feat = pd.read_csv(os.path.join('input', 'df_audio_feat.csv'), index_col=0)
            df_text_feat = pd.read_csv(os.path.join('input', 'df_text_feat.csv'), index_col=0)
            gv.logger.info('Data Loaded with success! ')

        gv.model_args = select_model(int(model))
        model_args = gv.model_args
        model_args['model_subDir'] += '_' + compress
        gv.logger.info('Starting Popularity Training Procedure!')
        output = dm.training_popularity_model(df_tracks, df_artists=df_artists,df_albums=None,
                                              df_audio_features=df_audio_feat,
                                              df_text_features=df_text_feat,
                                              model_args=model_args, mode='Train',
                                              task=task,
                                              new_preprocessing=new_preprocessing,
                                              compress=compress)
        if output:
            gv.logger.info('The Popularity Training Procedure was finished with success!')
    except Exception as e:
        gv.logger.error(e)
        output = {''}
    return output


def predict_musicPopNet(track_id):

    popularity = -1
    popularity_prob = [0, 0, 0]
    error = False
    output = {'Popularity Level': str(popularity),
              'Popularity Probabilities': popularity_prob,
              'Error': str(error),
              "Track Name": "Unknown",
              "Artist Names": "Unknown",
              "Preview URL": ""}

    try:
        gv.logger.info('Predicting popularity for Track %s', track_id)

        # Check data in the database
        x_test = gv.x_test

        try:
            track = x_test.loc[x_test['track_id'] == track_id]
            # remove track_id column
            track = track.drop(['track_id'], axis=1, inplace=False)

            track_info = gv.data_manager.get_track_artist_name(track_id=track_id)
            output['Track Name'] = track_info['track_name']
            output['Artist Names'] = track_info['artist_name']
            output['Preview URL'] = track_info['preview_url']
        except Exception as e:
            gv.logger.error(e)
            track = None

        if track is not None:
            # Get Name and Artist Name
            # Load Classification Model
            # Classification
            model = gv.musicPopNet_cls

            # Predict popularity
            with gv.graph.as_default():
                popularity_prob = model.predict(track)
                popularity_prob_ls = np.round(popularity_prob[0].tolist(), decimals=4)
                popularity = np.argmax(popularity_prob_ls,axis=-1)

        if popularity != -1:
            output['Popularity Probabilities'] = dict(zip(['Low', 'Medium', 'High'], [round(x,3) for x in popularity_prob_ls]))
            if popularity == 0:
                output['Popularity Level'] = 'Low'
            elif popularity == 1:
                output['Popularity Level'] = 'Medium'
            else:
                output['Popularity Level'] = 'High'

    except Exception as e:
        gv.logger.error(e)
        output['Error'] = e

    return output