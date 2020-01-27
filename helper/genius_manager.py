# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 10:54:17 2018

@author: dmarg
"""

from helper.data_helper import read_credentials
import lyricsgenius as genius
import requests
import re
from bs4 import BeautifulSoup
from helper import global_variables as gv

class GeniusManager():
    def __init__(self, config_file):
        self.config_file = config_file
        self.cid, self.secret, self.access_token = self.get_credentials()
        self.genius =  genius.Genius(self.access_token)
        self.response = None
        self.remote_song_info = None
        self.track_url = None
        self.artist_name = None
        self.track_name = None
        self.logger = gv.logger
        self.set_up_genius()
        
    def set_up_genius(self):
        ok=False
        try:
            self.genius.verbose = False # Turn off status messages
            self.genius.remove_section_headers = True # Remove section headers (e.g. [Chorus]) from lyrics when searching
            self.genius.skip_non_songs = True # Include hits thought to be non-songs (e.g. track lists)
            self.genius.replace_default_terms = True
            self.excluded_terms = ["(Remix)", "(Live)", "(Cover)",
                                   " - extended version",
                                   " - edit",
                                   "[explicit]",
                                   "remix"] # Exclude songs with these words in their title
            self.base_url = 'https://api.genius.com'
            self.headers = {'Authorization': 'Bearer ' + self.access_token}
            self.search_url = self.base_url + '/search'
            self.logger.info('Connected to Genius API')
            ok=True
        except Exception as e:
            self.logger.error(e)
        return ok
    
    def search_lyrics_genius(self, title, artist):
        lyrics = '-99'
        try:
            data = {'q': title + ' ' + artist}
            self.response = requests.get(self.search_url, 
                                         data=data, 
                                         headers=self.headers)
            
            lyric_content = self.response.json()
            for hit in lyric_content['response']['hits']:
                if artist.lower() in hit['result']['primary_artist']['name'].lower():
                    self.remote_song_info = hit
                    break
            # Extract lyrics from URL if the song was found
            if self.remote_song_info:
                self.track_url = self.remote_song_info['result']['url']
                # Get Lyric
                lyrics = self.scrap_song_url()
        except Exception as e:
            self.logger.error(e)
        return lyrics
    
    def scrap_song_url(self):
        try:
            page = requests.get(self.track_url)
            html = BeautifulSoup(page.text, 'html.parser')
            lyrics = html.find('div', class_='lyrics').get_text()
            
            # Remove headers
            lyrics = re.sub('(\[.*?\])*', '', lyrics)
            lyrics = re.sub('\n{2}', '\n', lyrics)  # Gaps between verses
        except Exception as e:
            self.logger.error(e)
        return lyrics

    def get_credentials(self):
        try:
            gen_cred = read_credentials(self.config_file, platform='GENIUS')
            cid, secret, acc_token = gen_cred
        except Exception as e:
            self.logger.error(e)
        return cid, secret, acc_token
    
    def get_track_data(self, artist_name, track_name):
        ok=False
        try:
            self.track_name = track_name
            artist = self.genius.search_artist(artist_name, max_songs=1,
                                               sort="title",
                                               get_full_info=False)
            if artist is not None:
                self.artist_name=artist.name
            ok=True
        except Exception as e:
            self.logger.error(e)
        return ok
    
    def get_lyrics(self, row, df_artist_data, threshold=0.6):
        lyrics = '-99'
        row = row.to_dict()
        try:
            instrumentalness = row['instrumentalness']
            name = row['name']
            track_id = row['id']
            if instrumentalness<=threshold:
                data_artist_row = df_artist_data.loc[df_artist_data['track_id'] == str(track_id)]
                self.track_name = str(name).lower().replace("'", "").replace("&", " and ")
                # Remove feat
                ft = '(feat'
                if ft in self.track_name:
                    idx = self.track_name.find(ft)
                    self.track_name = self.track_name[0:idx-1]
                
                # Clean Elements
                for exclude_el in self.excluded_terms:
                    self.track_name.replace(exclude_el, '')
               
                self.artist_name = data_artist_row['name'].values[0].replace("&", " and ").title()
                # Check if contains lyrics 
                if self.track_name and self.artist_name is not None:
                    self.logger.info('Searching Lyric in Genius ... "%s" by %s', self.track_name,
                          self.artist_name)
                    lyrics = self.search_lyrics_genius(title=self.track_name,
                                                       artist=self.artist_name)
                    if lyrics!='-99':
                        self.logger.info('Found!')
        except Exception as e:
            self.logger.error(e)
        return lyrics
    
    def process_data(self, df_tracks, df_artists, threshold=0.6, ):
        data_tracks = df_tracks.copy()
        try:
            if data_tracks is not None and df_artists is not None:
                self.logger.info('Collecting Lyrics from Genius ... ')
                #data_tracks['lyrics'] = data_tracks.swifter.apply(lambda row : self.get_lyrics(row, data_artists, threshold),axis=1)
                data_tracks['lyrics'] = data_tracks.apply(lambda x: self.get_lyrics(x, df_artists, threshold),axis=1)
        
        except Exception as e:
            self.logger.error(e)

        return data_tracks