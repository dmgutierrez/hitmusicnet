# -*- coding: utf-8 -*-

from helper.spotify_manager import SpotifyManager
from helper.genius_manager import GeniusManager
import pandas as pd
import numpy as np
import os
import glob
import subprocess
from helper import global_variables as gv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk,scan
from analysis.low_level_feature_extraction import AudioFeatures
from analysis.text_feature_extraction import TextFeatures
from analysis.popularity_prediction import PopularityPrediction


class DataManager:
    def __init__(self, config_file, host, port, countries):

        self.config_file = config_file
        self.playlists = None
        self.country_playlists = None
        self.set_playlists_id = None
        self.host = host
        self.port = port
        self.audio_dir = os.getenv('Audio_dir')
        self.es = None
        self.countries = countries
        self.sp = SpotifyManager(self.config_file)
        self.gen = GeniusManager(self.config_file)
        self.connect_to_elasticsearch()
        self.es_playlist_index = gv.indexNames

    def connect_to_elasticsearch(self):
        try:
            self.es = Elasticsearch([{'host': self.host, 'port': self.port}],
                                    http_compress=True)
            if self.es.ping():
                gv.logger.info("Connected to ElasticSearch at %s: %s", self.host, self.port)
            else:
                gv.logger.error("It was not possible to connect to %s: %s", self.host, self.port)
                ok=False
        except Exception as e:
            gv.logger.error(e)
        return self

    def retrieve_playlists_from_Spotify(self):
        try:
            gv.logger.info('Gathering playlists uuid\'s from Spotify ...')
            ok = self.sp.get_playlists_by_category(countries=self.countries)
            if ok:
                self.set_playlists_id = self.sp.playlists_id
                self.country_playlists = self.sp.country_playlists
        except Exception as e:
            gv.logger.error(e)
        return self
    
    def ingest_playlists_into_elastic(self):
        try:
            if self.set_playlists_id is not None:
                for i, pl_id in enumerate(self.set_playlists_id):
                    gv.logger.info("Processing Playlist %s/%s", i+1, len(self.set_playlists_id))
                    country_pl = self.country_playlists[i]
                    try:
                        track_data, artist_data, album_data =  self.sp.extract_data_from_playlist(pl_id,country_pl)
                        if track_data is not None and artist_data is not None and album_data is not None:
                            df_track_data = pd.DataFrame(track_data)
                            df_artist_data = pd.DataFrame(artist_data)
                            df_album_data = pd.DataFrame(album_data)
                            
                            # Add lyrics
                            df_track_data['lyrics'] = df_track_data.apply(lambda x: self.gen.get_lyrics(x, df_artist_data,
                                         threshold=0.6),axis=1)
                            data = [df_track_data, 
                                    df_artist_data,
                                    df_album_data]
                            
                            for i, df in enumerate(data):
                                if not self.es.indices.exists(index=self.es_playlist_index[i]):
                                    # Simple index creation with no particular mapping
                                    self.es.indices.create(index=self.es_playlist_index[i],body={})
                                
                                ok = self.ingest_data_into_elastic(df, indexName=self.es_playlist_index[i],
                                                                   doc_type=gv.doc_type[i])
                    except Exception as e:
                        gv.logger.error(e)
                        gv.logger.warning('Playlist not indexed because of an error')
                        continue                    
        except Exception as e:
            gv.logger.error(e)
        return self
    
    def generate_playlist_db(self):
        try:
            self.retrieve_playlists_from_Spotify()
            self.ingest_playlists_into_elastic()
            ok = True
        except Exception as e:
            gv.logger.error(e)
            ok = False
        return ok

    def extract_data_from_elasticsearch(self, index, doc_type, query):
        data=None
        try:
            gv.logger.info("Extracting data from elasticSearch index %s ...", index)
            # 1) Get total number of elements
            items = scan(self.es, query=query, index=index, doc_type=doc_type)
            data = self.parse_data(items)
        except Exception as e:
            gv.logger.error(e)
        return data
    
    def extract_info_item(self, item):
        item_dict = {}
        item_dict['source'] = item['_source']
        return item_dict

    def parse_data(self, items):
        df_data = None
        data = []
        try:
            for item in items:
                data.append(item['_source'])
            
            # Create two DataFrames
            df_data = pd.DataFrame(data)
            # Concatenate DataFrames and add the ElasticSearch Index
            #df_data = pd.concat([df_source, df_info], axis=1)
        except Exception as e:
            gv.logger.error(e)
        
        return df_data
    
    def ingest_data_into_elastic(self, df_spotify, indexName, doc_type, index_col='id'):
        ok = False
        try:
            if 'type' in list(df_spotify):
                df_spotify.drop('type', axis=1, inplace=True)
            df_spotify.drop_duplicates(subset=[index_col], keep="first", 
                                       inplace=True)
            
            df_spotify.fillna(value=np.nan, inplace=True) 
            df_spotify.fillna(value='', inplace=True)
            df_spotify = df_spotify.astype({index_col: str})
            df_spotify['track_name_prev'] = ['track_' + str(x+1) for x in range(len(df_spotify))]
            
            if df_spotify is not None:
                self.bulk_data_ES(df_spotify, indexName, doc_type, index_col)
        except Exception as e:
            gv.logger.error(e)
        return ok
    
    def bulk_data_ES(self, df, indexName, doc_type, index_col):
        try:
            def generate_dict(df):
                records = df.to_dict(orient='records')
                for record in records:
                    yield record
            
            if df is not None:
                # Bulk inserting documents. Each row in the DataFrame will be a document in ElasticSearch
                actions = ({'_index': indexName,
                              '_type': doc_type,
                              '_id': record[index_col],
                              '_source': record}
                                for record in generate_dict(df))
                gv.logger.info("Inserting Data into Elasticsearch at index: %s", indexName)
                bulk(self.es, actions, refresh='wait_for', raise_on_error=True)

        except Exception as e:
            gv.logger.error(e)
        return self
    
    def remove_ES_index(self, es_index):
        ok=False
        try:
            self.es.indices.delete(index=es_index, ignore=[400, 404])
            ok=True
        except Exception as e:
            gv.logger.error(e)
        return ok
    
    def prepare_directory(self):
        ok = False
        try:
            if not os.path.exists(self.audio_dir):
                os.makedirs(self.audio_dir)
            ok=True
        except Exception as e:
            gv.logger.error(e)
        return ok
    
    def download_file(self, url, audio_path, audio_name):
        ok=False
        try:
            audio_path = os.path.join(audio_path, audio_name)
            if url is not None and audio_path is not None:
                gv.logger.info('Downloading and storing File: %s', audio_name)
                request = 'Curl ' + url + ' -o ' + audio_path 
                subprocess.call(request)
                ok=True
        except Exception as e:
            gv.logger.error(e)
        return ok
    
    def collect_low_level_features_ES(self, df, stat='mean', index_name=None,
                                      doc_type=None, index_col='track_id'):
        ok=False
        self.prepare_directory()
        n = len(df)
        # ES Index
        if not self.es.indices.exists(index=index_name):
            gv.logger.info('Creating New Index in ES: %s', index_name)
            self.es.indices.create(index=index_name, body={})
        for index, row in df.iterrows():
            try:
                print(80*'=')
                gv.logger.info('......... Extracting Audio from index %s/%s .........', index+1, n)
                print(80*'=')
                preview_audio_url = row["preview_url"]
                if preview_audio_url is not None:
                    # Download File
                    url = preview_audio_url
                    audio_name = 'track' + '_' + str(row["id"]) + '.mp3'
                    done = self.download_file(url, self.audio_dir, audio_name)
                    if done:
                        gv.logger.info('Extracting Low-level features ...')
                        # Collect features
                        audioFeat = AudioFeatures(self.audio_dir, audio_name)
                        ok = audioFeat.run(index=0, stat='mean')
                        if ok:
                            # Bulk into ES
                            df = audioFeat.df_low_level_features.copy()
                            df[index_col] = row['id']
                            self.bulk_data_ES(df, index_name, doc_type, index_col)
            except Exception as e:
                gv.logger.error(e)
                continue
        return ok
    
    def collect_text_features_ES(self, df, index_name=None,doc_type=None,
                                 index_col='track_id',max_cum=.80, mu=.40):
        ok=False
        n = len(df)
        # ES Index
        if not self.es.indices.exists(index=index_name):
            gv.logger.info('Creating New Index in ES: %s', index_name)
            self.es.indices.create(index=index_name,body={})
        for index, row in df.iterrows():
            try:
                print(80*'=')
                gv.logger.info("......... Extracting Lyrics from index %s/%s .........", index+1, n)
                print(80*'=')
                lyrics = row["lyrics"]
                
                gv.logger.info('Extracting Low-level features ...')
                # Collect text features
                textFeat = TextFeatures(lyrics)
                ok = textFeat.run(index=0, max_cum=max_cum, mu=mu)
                if ok:
                    # Bulk into ES
                    df = textFeat.df_text_features.copy()
                    df[index_col] = row['id']
                    self.bulk_data_ES(df, index_name, doc_type, index_col)
            except Exception as e:
                gv.logger.error(e)
                continue
        return ok

    def training_popularity_model(self, df_tracks, df_artists, df_albums, df_audio_features,
                                  df_text_features, model_args, mode='Train', task='regression',
                                  new_preprocessing=True, compress='auto'):
        try:
            pop_prediction = PopularityPrediction(df_tracks, df_artists, df_albums, df_audio_features,
                                                  df_text_features, model_args, mode=mode, task=task,
                                                  new_preprocessing=new_preprocessing, compress=compress)
            output = pop_prediction.run()

        except Exception as e:
            gv.logger.error(e)
            output = {''}
        return output

    def ingest_samples_ES(self):
        path = self.audio_dir + os.sep + '*.mp3'
        audio_files = glob.iglob(path, recursive=True)
        index_name = 'audio_samples'
        doc_type = 'Samples'
        ok=True
        for i, audio_name in enumerate(audio_files):
            try:
                # Get Track ID
                track_id = os.path.basename(os.path.normpath(audio_name))
                track_id = track_id.replace('track_', '').replace('.mp3', '')
                # Retrieve Popularity from document in ES
                res = self.es.get(index='spotify_tracks', doc_type='Track', id=track_id)
                if res:
                    y = res['_source']['popularity']
                    # Load audio file
                    audio_feat = AudioFeatures(self.audio_dir, audio_name)
                    x = audio_feat.load_audio().tolist()
                    data = {'track_id': track_id, 'samples': [x], 'popularity': y}
                    df = pd.DataFrame(data, index=[0])
                    # Put data into ElasticSearch
                    # ES Index
                    if not self.es.indices.exists(index=index_name):
                        gv.logger.info("Creating New Index in ES: %s", index_name)
                        self.es.indices.create(index=index_name, body={})

                    self.bulk_data_ES(df, indexName=index_name, doc_type=doc_type, index_col='track_id')
            except Exception as e:
                gv.logger.error(e)
                ok = False
                continue
        return ok

    def get_track_artist_name(self, track_id):
        output = {"track_name":  "Unknown",
                  "artist_name": ["Unknown"],
                  "preview_url": ""}
        try:
            gv.logger.info('Collecting Track Name and Artists Name')
            track = self.sp.sp.track(track_id=track_id)
            output["track_name"] = track['name']

            # Add Preview
            prev = track['preview_url']
            if prev is not None:
                output['preview_url'] = prev

            # Artists
            artist_names = []
            all_artists = track['artists']
            for art in all_artists:
                artist_names.append(art['name'])

            output["artist_name"] = artist_names
        except Exception as e:
            gv.logger.error(e)
        return output