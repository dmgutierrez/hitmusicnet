# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
from functools import reduce
from sklearn.model_selection import StratifiedKFold,KFold, train_test_split
from models.dnn_models import DeepLearningModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from helper import global_variables as gv
from sklearn.decomposition import SparsePCA
from sklearn.metrics import classification_report


class PopularityPrediction:
    def __init__(self, df_tracks, df_artists, df_albums, df_audio_features,
                 df_text_features, model_args, mode='Train', task='regression',
                 new_preprocessing=True, compress='auto'):

        self.logger = gv.logger
        self.df_tracks = df_tracks
        self.df_artists = df_artists
        self.df_albums = df_albums
        self.df_audio_features = df_audio_features
        self.df_text_features = df_text_features
        self.model_args = model_args
        self.mode = mode
        self.task = task
        self.new_preprocessing = new_preprocessing
        self.compress = compress
        if self.task == 'regression':
            self.target = 'popularity'
        else:
            self.target = 'popularity_class'
        self.mode = mode
        self.X = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.X_sc = None
        self.y_sc = None
        self.scaler = None
        self.dl_model = None
        self.model_history_cv = {}
        self.output = {}
        self.set_up_model()

    def get_popularity_class(self, row, scale=1):
        pop_class = -1
        try:
            if row['popularity'] < 25*scale:
                pop_class = 0
            elif row['popularity'] >= 25*scale and row['popularity'] < 65 :
                pop_class = 1
            else:
                pop_class = 2
                
        except Exception as e:
            self.logger.error(e)
        return pop_class

    def get_artist_data(self, row, col_name):
        final_art_data = -99
        try:

            artist_id_n = row['artists_id'][2:-2].replace("'", "").split() # Remove brackets
            artist_id = [a_id.replace(",", "") for a_id in artist_id_n]

            out = []
            # Yields Artist data
            for id in artist_id:
                res = self.df_artists[self.df_artists['id'] == id]
                if res.shape[0]>0:
                    art_data = res[col_name].values[0]
                    out.append(art_data)
            final_art_data = int(np.mean(out))
        except Exception as e:
            self.logger.error(e)
        return final_art_data

    def get_total_markets(self, row):
        n_markets = 0
        try:
            n_markets = len(row['available_markets'][2:-2].replace("'", "").split())
        except Exception as e:
            print(e)
        return n_markets

    def data_preprocessing(self, compressed=True, compressed_method='AE',save_data=True):
        ok=True
        try:
            self.logger.info('Pre-processing data for %s problem ...', self.task)
            if self.new_preprocessing:
                key_id = 'track_id'
                # Rename id column to be consistent in all DataFrames
                self.df_tracks = self.df_tracks.rename(columns={'id': key_id})

                # Adding the popularity of the artist
                if self.df_artists is not None:
                    # Apply transformation
                    self.logger.info('Applying Market transformation ...')
                    self.df_tracks['n_markets'] = self.df_tracks.apply(lambda x: self.get_total_markets(x), axis=1)
                    self.logger.info('Applying Artists followers transformation ...')
                    self.df_tracks['artist_followers'] = self.df_tracks.apply(lambda x:
                                                                              self.get_artist_data(x, 'followers'),
                                                                              axis=1)
                    self.logger.info('Applying Artists popularity transformation ...')
                    self.df_tracks['artist_popularity'] = self.df_tracks.apply(lambda x:
                                                                               self.get_artist_data(x, 'artist_popularity'),
                                                                               axis=1)

                # Join DataFrames
                df_join = [self.df_tracks, self.df_audio_features,
                           self.df_text_features]
                # --------------------------------------------------------------------
                # Merge Data by id
                df_final = reduce(lambda left, right: pd.merge(left, right, on=key_id),
                                  df_join)

                # .......................
                # --------------------------------------------
                tracks_id = df_final[['track_id']]
                tracks_id.to_csv(os.path.join('input', 'tracks_id_' + self.task + '.csv'))
                # .......................
                # Drop cols
                drop_cols = ['album_id', 'analysis_url', 'artists_id',
                             'available_markets', 'country', 'disc_number',
                             'lyrics','name', 'playlist','preview_url',
                             'track_href', 'track_name_prev', 'track_number',
                             'uri','time_signature','href', 'track_id', 'type']

                # Add classification target
                if self.task == 'classification':
                    df_final[self.target] = df_final.apply(self.get_popularity_class, scale=1, axis=1)
                    # Remove regression col
                    drop_cols += ['popularity']

                check_cols = [col for col in drop_cols if col in list(df_final.columns)]

                df_final.drop(check_cols, axis=1, inplace=True)

                # ----------------------------------------
                # Predictor-Response Variables
                self.y = df_final.loc[:, self.target]
                self.X = df_final.drop(self.target, axis=1)

                # Scale Data
                self.scaler = MinMaxScaler(feature_range=(0, 1))
                self.X_sc = self.scaler.fit_transform(self.X)
                if self.task == 'regression':
                    self.y_sc = self.scaler.fit_transform(np.array(self.y).reshape(-1, 1))
                    self.y_sc = pd.DataFrame(self.y_sc, columns=[self.target])
                else:
                    self.y_sc = pd.DataFrame(self.y, columns=[self.target])
                # Convert to DataFrame
                self.X_sc = pd.DataFrame(self.X_sc, columns=self.X.columns)

                if save_data:
                    x_save = pd.concat([self.X_sc, self.y_sc], axis=1)
                    x_save.to_csv(os.path.join('input', 'original_signal.csv'))
                    self.y_sc.to_csv(os.path.join('input', 'labels.csv'))

                # Apply Compressed Autoencoder
                if compressed:
                    if compressed_method == 'SparsePCA':
                        self.logger.info('Applying Sparse PCA to reduce the dimensionality ... ')
                        n_comp = int(self.X_sc.shape[1]/5)
                        transformer = SparsePCA(n_components=n_comp,
                                                random_state=0)
                        x_compressed = transformer.transform(self.X_sc)

                    else:
                        self.logger.info('Applying Autoencoder to reduce the dimensionality ... ')
                        x_compressed = self.apply_feature_compression()
                        # Scale feature vector from autoencoder
                        x_compressed = self.scaler.fit_transform(x_compressed)

                    cols = ['feat_compressed_' + str(i+1) for i in range(x_compressed.shape[1])]
                    self.X_sc = pd.DataFrame(x_compressed, columns=cols)

                    # Saving Compressed Data
                    if save_data:
                        x_compr = pd.concat([self.X_sc, self.y_sc], axis=1)
                        name_comp = compressed_method + '_compressed_' + self.compress + '.csv'
                        x_compr.to_csv(os.path.join('input', name_comp))

            # LOAD DATA
            else:
                name = compressed_method + '_compressed_' + self.compress + '.csv'
                self.logger.info('Loading Compressed Data from local file ... %s', name)
                self.X_sc = pd.read_csv(os.path.join('input', name), index_col=0)
                # Remove Target
                if self.target in list(self.X_sc.columns):
                    self.X_sc.drop([self.target], axis=1, inplace=True)
                self.y_sc = pd.read_csv(os.path.join('input', 'labels.csv'), index_col=0)
        except Exception as e:
            self.logger.error(e)
            ok = False
        return ok
    
    def set_up_model(self):
        ok = True
        try:
            self.dl_model = DeepLearningModel(model_name=self.model_args['model_name'],
                                              model_dir=self.model_args['model_dir'],
                                              model_subDir=self.model_args['model_subDir'],
                                              input_dim=self.model_args['input_dim'],
                                              output_dim=self.model_args['output_dim'],
                                              optimizer=self.model_args['optimizer'],
                                              metrics=self.model_args['metrics'],
                                              loss=self.model_args['loss'],
                                              add_earlyStopping=self.model_args['earlyStop'],
                                              weights_path=self.model_args['weights'],
                                              plot_loss=self.model_args['plot_loss'],
                                              neuron_parameters=self.model_args['neurons'],
                                              layers=self.model_args['n_layers'],
                                              initialization=self.model_args['weights_init'],
                                              level_dropout=self.model_args['dropout'],
                                              problem=self.task)
        except Exception as e:
            self.logger.error(e)
            ok = False
        return ok
    
    def train_model(self, n_splits=5, shuffle_data=True):
        ok = True
        try:

            # --------------------------------------------
            tracks_id = pd.read_csv(os.path.join('input', 'tracks_id_' + self.task + '.csv'),
                                    index_col=0)

            # Separate into Train and Test
            x_total = pd.concat([self.X_sc, self.y_sc, tracks_id], axis=1)
            train, test = train_test_split(x_total, test_size=0.10)

            self.y_train = train.loc[:, self.target]
            self.x_train = train.drop([self.target], axis=1, inplace=False)
            self.y_test = test.loc[:, self.target]
            self.x_test = test.drop([self.target], axis=1, inplace=False)

            # Save test
            self.x_test.to_csv(os.path.join('input', 'x_test_' + self.task + '.csv'))
            self.y_test.to_csv(os.path.join('input', 'y_test_' + self.task + '.csv'))

            # Remove ids
            self.x_train.drop(['track_id'], axis=1, inplace=True)
            self.x_test.drop(['track_id'], axis=1, inplace=True)
            # ----------------------------------------------

            # 1) Choose Cross-validation method
            if self.task == 'classification':
                skf = StratifiedKFold(n_splits=n_splits)
                splits = skf.split(self.x_train, pd.DataFrame(self.y_train, columns=[self.target]))
            else:
                skf = KFold(n_splits=n_splits, shuffle=shuffle_data)
                splits = skf.split(self.x_train.index)

            if self.dl_model is None:
                self.set_up_model()

            # Change input Dimension and build model
            self.dl_model.input_dim = self.x_train.shape[1]
            done = self.dl_model.build_model()

            if done:
                start = time.time()
                for index, (train_indices, val_indices) in enumerate(splits):
                    self.logger.info("Training on fold %s/%s", index+1, n_splits)

                    # Generate batches from indices
                    x_train, x_val = np.array(self.x_train.iloc[train_indices]).round(3),\
                                     np.array(self.x_train.iloc[val_indices]).round(3)
                    y_train, y_val = np.array(self.y_train.iloc[train_indices]).round(3),\
                                     np.array(self.y_train.iloc[val_indices]).round(3)

                    # Compile the model
                    self.dl_model.compile_model()

                    # Train the DL model
                    model_history = self.dl_model.train_model(x_train, y_train,
                                                              x_val, y_val,
                                                              epochs=self.model_args['epochs'],
                                                              batch_size=self.model_args['batch_size'],
                                                              monitor_early=self.model_args['early_mon'],
                                                              mode=self.model_args['mode'],
                                                              monitor_checkout=self.model_args['checkout_mon'])
                    self.logger.info('Saving Model History ... ')
                    history = model_history.history
                    # Save Metrics
                    for metric in list(history.keys()):
                        self.model_history_cv[metric] = history[metric][-1]

                    # Save Model and History
                    self.dl_model.save_model()
                    self.dl_model.save_history(model_history, k=index)
                    ok = True

                end = time.time()

                self.model_history_cv['cpu (ms)'] = end-start

                # Predict Function
                self.logger.info('Calculating predictions using x_test ... ')
                y_pred = self.dl_model.validate_model(X_test=self.x_test)

                if self.task == 'regression':
                    mae_test = mean_absolute_error(y_true=np.array(self.y_test), y_pred=y_pred)
                    self.model_history_cv['test_mean_absolute_error'] = mae_test
                else:
                    # Convert one-hot to index
                    y_pred = np.argmax(y_pred, axis=1).transpose()
                    target_names = ['Low', 'Medium', 'High']
                    self.model_history_cv['report'] = classification_report(y_true=np.array(self.y_test),
                                                                            y_pred=y_pred,
                                                                            target_names=target_names,
                                                                            output_dict=True)
                self.logger.info('Writing Model Summary ... ')
                # Summary results
                self.output = self.dl_model.get_model_summary(history=self.model_history_cv, task=self.task,
                                                              k_fold=n_splits)
            else:
                ok = False
        except Exception as e:
            self.logger.error(e)
            ok = False
        return ok
    
    def apply_feature_compression(self):
        x_compressed = None
        try:
            encoder = self.dl_model.train_autoencoder(x_train=self.X_sc, epochs=50,
                                                      compress=self.compress)
            x_compressed = encoder.predict(self.X_sc)
        except Exception as e:
            self.logger.error(e)
        return x_compressed

    def predict(self):
        try:
            pass
        except Exception as e:
            self.logger.error(e)
        return
    
    def run(self):
        output = {''}
        try:
            # 1) Train the Neural Network
            if self.mode == 'Train':
                # 1) Data Pre-processing
                ok = self.data_preprocessing()
                if ok:
                    ok = self.train_model()
                    output = self.output
            # Predict
            else:
                # Load Model and predict
                pass
        except Exception as e:
            self.logger.error(e)
            output = {''}

        return output