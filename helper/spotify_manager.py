# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 19:45:03 2018

@author: dmarg
"""
import spotipy
from spotipy.client import SpotifyException
from spotipy.oauth2 import SpotifyClientCredentials
from helper.data_helper import read_credentials
import datetime
import time
from helper import global_variables as gv
# =============================================================================
class SpotifyManager():
    def __init__(self, config_file):
        self.config_file = config_file
        self.cid, self.secret, self.username = self.get_credentials()
        self.client_credentials_manager = None
        self.token = None
        self.token_info=None
        self.scope = 'user-library-read'
        self.redirect_uri = 'https://song-popularity-analyzer.com/callback/'
        self.sp = None
        self.logger = gv.logger
        self.set_up_aunthentication()
        self.sp.trace=False
        self.categories_raw = None
        self.playlists_id = []
        self.country_playlists = []

        
    def get_credentials(self):
        spotify_cred = read_credentials(self.config_file, platform='SPOTIFY')
        cid, secret, username = spotify_cred
        return cid, secret, username
    
    def is_token_expired(self):
        expired=False
        try:
            now = int(time.time())
            expired=self.token_info['expires_at'] - now < 60
        except SpotifyException as e:
            self.logger.error(e)
        return expired
    
    def set_up_aunthentication(self):
        ok=False
        try:
            self.client_credentials_manager = SpotifyClientCredentials(client_id=self.cid,
                                                                       client_secret=self.secret)
            self.token = self.client_credentials_manager.get_access_token()
            self.token_info = self.client_credentials_manager.token_info
            ok=True
            self.sp =spotipy.Spotify(client_credentials_manager=self.client_credentials_manager)
            self.logger.info('Connected to Spotify API')
        except SpotifyException as e:
            self.logger.error(e)
            ok=False
        return ok
    
    def get_categories_id(self, country ='ES'):
        categories_items = None
        try:
            self.categories_raw  = self.sp.categories(limit=50, offset=0, country=country)
            categories_items = self.categories_raw['categories']['items']
        except Exception as e:
            self.logger.error(e)
        return categories_items
    
    def get_playlist(self, playlist_id):
        try:
            playlist = self.sp.user_playlist(self.username, playlist_id)
        except Exception:
            playlist = None
        return playlist
    
    def show_playlist_data(self, playlist):
        if playlist['owner']['id'] == self.username:
            print('')
            print(playlist['name'])
            print('  total tracks', playlist['tracks']['total'])
            results = self.sp.user_playlist(self.username, playlist['id'], fields="tracks,next")
            tracks = results['tracks']
            self.show_tracks(tracks)
            while tracks['next']:
                tracks = self.sp.next(tracks)
                self.show_tracks(tracks)
        return tracks
    
    def get_playlists_by_category(self, countries):
        playlists = []
        ok=False
        try:
            for idx, i in enumerate(countries):
                self.logger.info("Collecting Playlists from %s, %s/%s", i, idx+1, len(countries))
                # Get Category ID
                category_items = self.get_categories_id(country=i)
                # Get Playlists
                data = [self.sp.category_playlists(category_id=c['id'], country=i,
                                                   limit=50, offset=0) for c in category_items if not c['id']=='audiobooks']
                playlists += data
                # Remove duplicates
                self.collect_id_from_playlists(playlists)
                self.playlists_id = list(set(self.playlists_id))
                self.country_playlists += [i for ct in range(len(self.playlists_id))]
            ok=True
        except Exception as e:
            self.logger.error(e)
        return ok
    
    
    def collect_id_from_playlists(self, playlists):
        ok = True
        try:
            for pl in playlists:
                items = pl['playlists']['items']
                self.playlists_id += [i['id'] for i in items]
        except Exception as e:
            self.logger.error(e)
            ok = False
        return ok
    
    def extract_data_from_playlist(self, playlist_id, country_pl):
        track_data = []
        artist_data = []
        album_data = []
        if not self.is_token_expired():
            try:
                playlist = self.get_playlist(playlist_id)
                playlist_name = playlist["name"]            
                playlist_tracks = self.sp.user_playlist_tracks(self.username, playlist_id)
                songs = playlist_tracks["items"]
                for j in range(len(songs)):
                    try:
                        self.logger.info("Processing Track ... %s/%s", j+1, len(songs))
                        track = songs[j]["track"] 
                        track_id = track["id"]
                        preview = track['preview_url']
                        if track is not None and track_id is not None and preview is not None:
                            tr_dt = self.collect_track_information(track, playlist_name, country_pl)
                            al_dt = self.collect_album_information(track)
                            ar_dt = self.collect_artist_information(track)
                            if tr_dt is not None and al_dt is not None and ar_dt is not None:
                                track_data.append(tr_dt)
                                album_data.append(al_dt)
                                artist_data += ar_dt
                    except:
                        self.logger.warning('It was not possible to extract the information of the song!')
                        continue
            except Exception as e:
                self.logger.error(e)
        else:
            self.set_up_aunthentication()
            self.extract_data_from_playlist(playlist_id)
        return track_data, artist_data, album_data
    
    def collect_track_information(self, track, playlistName, country_pl):
        track_data = {}
        try:
            audio_features = self.collect_audio_features(track)
            if audio_features is not None:
                track_data['id'] = track['id']
                track_data['name'] = track['name']
                track_data['popularity'] = track['popularity']
                track_data['preview_url'] = track['preview_url']
                track_data['available_markets'] = track['available_markets']
                track_data['disc_number'] = track['disc_number']
                track_data['duration_ms'] = track['duration_ms']
                track_data['href'] = track['href']
                track_data['track_number'] = track['track_number']
                track_data['type'] = track['type']
                track_data['artists_id'] = [art['id'] for art in track['artists']]
                track_data['album_id'] = track['album']['id']
                track_data['playlist'] = playlistName
                track_data['country'] = country_pl.upper()
                track_data.update(audio_features)
        except Exception as e:
            self.logger.error(e)
            return None
        return track_data
    
    def collect_audio_features(self, track):
        audio_features = None
        try:
            track_id = track['id']
            audio_features = self.sp.audio_features(track_id)[0]
            audio_features.pop('id', None)
        except Exception as e:
            self.logger.error(e)
            return None
        return audio_features
    
    def collect_album_information(self, track):
        album_data = {}
        try:
            track_id = track['id']
            album_data = track['album']
            artists_id = album_data['artists'][0]['id']
            album_data.pop('artists', None)
            album_data['track_id'] = track_id
            album_data['artist_id'] = artists_id
            
            if 'release_date' in list(album_data.keys()):
                if isinstance(album_data['release_date'], datetime.datetime):
                    date = album_data['release_date']
                    album_data['release_date'] = date.strftime('%Y-%m-%d')
            
        except Exception as e:
            self.logger.error(e)
            return None
        return album_data
    
    def collect_artist_information(self, track):
        artist_data = []
        try:
            track_id = track['id']
            artists = track['artists']
            artist_data += [self.get_artists_info(track_id=track_id, artist_id=i['id']) for i in artists]
        except Exception as e:
            self.logger.error(e)
            return None
        return artist_data
    
    def get_artists_info(self, track_id, artist_id):
        artist_data ={}
        try:
            data = self.sp.artist(artist_id)
            genres_info  = data['genres']
            followers = data["followers"]["total"]
            artist_data={'id':artist_id,
                        'name':data['name'],'genres':genres_info,
                         'artist_popularity': data['popularity'],
                         'followers': followers,
                         'track_id':track_id}
        except Exception as e:
            self.logger.error(e)
        return artist_data
# =============================================================================