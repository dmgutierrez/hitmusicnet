# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:51:19 2019

@author: dmarg
"""

import librosa as li
import numpy as np
import pandas as pd
import os
from pydub.utils import mediainfo


class AudioFeatures:
    def __init__(self, audio_dir, audio_filename):
        self.audio_dir = audio_dir
        self.audio_filename = audio_filename
        self.file_path = os.path.join(self.audio_dir, self.audio_filename)
        self.low_level_features = None
        self.mfcc_data = None
        self.cqt_data = None
        self.mel_data = None
        self.tonnetz_data = None
        self.spc_data = None
        self.df_low_level_features = None
        self.sr = self.get_sample_rate()
        self.y = self.load_audio()
        
    def get_sample_rate(self):
        sr = 44100
        try:
            audio_info = mediainfo(self.file_path)
            sr = int(audio_info['sample_rate'])
        except Exception as e:
            print(e)
        return sr
    
    def load_audio(self):
        y = None
        try:
            y, sr = li.load(self.file_path, sr=self.sr, mono=True)
        except Exception as e:
            print(e)
        return y
    
    def compute_mfcc(self, n_mfcc=48):
        mfcc = None
        try:
            mfcc = li.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=n_mfcc)
        except Exception as e:
            print(e)
        return mfcc
    
    def compute_chromagram_cqt(self, hop_length=512):
        chroma_cq = None
        try:
            chroma_cq = li.feature.chroma_cqt(y=self.y, sr=self.sr,
                                              hop_length=hop_length)
        except Exception as e:
            print(e)
        return chroma_cq
    
    def compute_mel_spectrogram(self):
        mel = None
        try:
            mel = li.feature.melspectrogram(y=self.y, sr=self.sr)
        except Exception as e:
            print(e)
        return mel
    
    def compute_zero_crossing_rate(self,frame_length=2048, hop_length=512):
        zcr = None
        try:
            zcr = li.feature.zero_crossing_rate(y=self.y, 
                                                frame_length=frame_length,
                                                hop_length=hop_length,
                                                center=True)
        except Exception as e:
            print(e)
        return zcr
    
    def compute_tonnetz(self):
        tonnetz = None
        try:
            tonnetz = li.feature.tonnetz(y=self.y, sr=self.sr)
        except Exception as e:
            print(e)
        return tonnetz
    

    def compute_spectral_centroid(self, n_fft=2048,hop_length=512):
        sc = None
        try:
            sc = li.feature.spectral_centroid(y=self.y, sr=self.sr,
                                              n_fft=n_fft,
                                              hop_length=hop_length)
        except Exception as e:
            print(e)
        return sc
    
    def compute_spectral_bandwdith(self, n_fft=2048,hop_length=512):
        sb = None
        try:
            sb = li.feature.spectral_bandwidth(y=self.y, sr=self.sr,
                                               n_fft=n_fft,
                                               hop_length=hop_length)
        except Exception as e:
            print(e)
        return sb
    
    def compute_spectral_contrast(self, n_fft=2048, hop_length=512):
        sc = None
        try:
            sc = li.feature.spectral_contrast(y=self.y,
                                              sr=self.sr,
                                              n_fft=n_fft, 
                                              hop_length=hop_length)
        except Exception as e:
            print(e)
        return sc
    
    def compute_spectral_rolloff(self, n_fft=2048, hop_length=512,
                                 roll_percent=0.85):
        sr = None
        try:
            sr = li.feature.spectral_rolloff(y=self.y,
                                             sr=self.sr,
                                             n_fft=n_fft, 
                                             hop_length=hop_length,
                                             roll_percent=roll_percent)
        except Exception as e:
            print(e)
        return sr
    
    def compute_energy_entropy(self):
        entropy = 0.0
        
        try:
            rmse = li.feature.rmse(self.y)
            mu_energy = np.mean(rmse)

            for j in range(0,len(rmse[0])):
                 q = float(np.absolute(rmse[0][j] - mu_energy))
                 entropy +=  (q * np.log10(q))

        except Exception as e:
            print(e)
        return entropy 
    
    def apply_stat(self, value, stat='mean'):
        if stat=='mean':
            res = np.mean(value)
        else:
            res = np.median(value)
        return res
    
    def collect_all_features(self, stat='mean', n_mfcc=48, hop_length=512,
                             frame_length=2048,n_fft=2048, 
                             roll_percent_min=0.1, roll_percent_max=0.85):
        ok = False
        try:
            self.low_level_features = {}
            # Collect features 
            # MFCC
            print('Computing MFCC ...')
            mfcc = self.compute_mfcc(n_mfcc=n_mfcc)
            mfcc_m = [self.apply_stat(mfcc[k],stat) for k in range(0, len(mfcc))]
            cols_mfccs = ['MFCC_' + str(i+1) for i in range(n_mfcc)]
            self.mfcc_data = dict(zip(cols_mfccs, mfcc_m))
            
            print('Computing CHROMA ...')
            # Chromagram
            cqt = self.compute_chromagram_cqt(hop_length=hop_length)
            cqt_m = [self.apply_stat(cqt[k], stat) for k in range(0, len(cqt))]
            cols_cqt = ['Chroma_' + str(i+1) for i in range(len(cqt_m))]
            self.cqt_data = dict(zip(cols_cqt, cqt_m))
            
            print('Computing MEL ...')
            # Mel
            mel = self.compute_mel_spectrogram()
            mel_m = [self.apply_stat(mel[k], stat) for k in range(len(mel))]
            cols_mel = ['MEL_' + str(i+1) for i in range(len(mel_m))]
            self.mel_data = dict(zip(cols_mel, mel_m))
            
            print('Computing ZCR ...')
            # ZCR
            zcr = self.compute_zero_crossing_rate(frame_length=frame_length,
                                                  hop_length=hop_length)
            self.low_level_features['ZCR'] = self.apply_stat(zcr, stat) 
            
            print('Computing TONNETZ ...')
            # Tonnetz
            tonnetz = self.compute_tonnetz()
            tonnetz_m = [self.apply_stat(tonnetz[k], stat) for k in range(0, len(tonnetz))]
            cols_ton = ['Tonnetz_' + str(i+1) for i in range(len(tonnetz_m))]
            self.tonnetz_data = dict(zip(cols_ton, tonnetz_m))
            
            print('Computing SP CENTROID ...')
            # Spectral Centroid 
            spc = self.compute_spectral_centroid(n_fft=n_fft,hop_length=hop_length)[0][0]
            self.low_level_features['spectral_centroid'] =  self.apply_stat(spc, stat) 
            
            print('Computing SP CONTRAST ...')
            # Spectral Contrast
            spcontr = self.compute_spectral_contrast(n_fft=n_fft,hop_length=hop_length)
            spcontr_m = [self.apply_stat(spcontr[k], stat) for k in range(0, len(spcontr))]
            cols_spec = ['Spectral_contrast_' + str(i+1) for i in range(len(spcontr_m))]
            self.spc_data = dict(zip(cols_spec, spcontr_m))
            
            print('Computing SP ROLLOFF ...')
            # Spectral Roll-off
            sproll_min = self.compute_spectral_rolloff(n_fft=n_fft,hop_length=hop_length,
                                                   roll_percent=roll_percent_min)
            sproll_max = self.compute_spectral_rolloff(n_fft=n_fft,hop_length=hop_length,
                                                       roll_percent=roll_percent_max)
            self.low_level_features['spectral_rollOff_min'] = self.apply_stat(sproll_min, stat) 
            self.low_level_features['spectral_rollOff_max'] = self.apply_stat(sproll_max, stat) 
            
            print('Computing BANDWIDTH ...')
            # Spectral Bandwidth
            spband = self.compute_spectral_bandwdith(n_fft=n_fft,hop_length=hop_length)
            self.low_level_features['spectral_bandwith'] = self.apply_stat(spband, stat) 
            
            print('Computing ENTROPY ...')
            # Entropy of the energy
            energy_entropy = self.compute_energy_entropy()
            self.low_level_features['entropy_energy'] = energy_entropy
            
            ok=True            
        except Exception as e:
            print(e)
        return ok
    
    def run(self, index, stat='mean', n_mfcc=48, hop_length=512,frame_length=2048,n_fft=2048,
            roll_percent_min=0.1, roll_percent_max=0.85):
        ok = False
        try:
            ok = self.collect_all_features(stat,n_mfcc, hop_length,
                                           frame_length, n_fft,
                                           roll_percent_min,
                                           roll_percent_max)
            if ok:
                # Create DataFrames and concat them
                df_ll = pd.DataFrame(self.low_level_features, index=[index])
                df_mfcc_data = pd.DataFrame(self.mfcc_data, index=[index])
                df_cqt_data = pd.DataFrame(self.cqt_data, index=[index])
                df_mel_data = pd.DataFrame(self.mel_data, index=[index])
                df_tonnetz_data = pd.DataFrame(self.tonnetz_data, index=[index])
                df_spc_data = pd.DataFrame(self.spc_data, index=[index])
                self.df_low_level_features = pd.concat([df_ll,
                                                        df_mfcc_data,
                                                        df_cqt_data,
                                                        df_mel_data,
                                                        df_tonnetz_data,
                                                        df_spc_data],axis=1)
        except Exception as e:
            print(e)
        return ok