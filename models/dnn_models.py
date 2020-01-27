# -*- coding: utf-8 -*-
from helper import global_variables as gv
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping,Callback, ModelCheckpoint, TensorBoard
from keras.models import Sequential, model_from_json, Model
from keras.utils import to_categorical
from models.classification_metrics_dnn import recall_m, precision_m, f1_m
from sklearn.utils import class_weight
import os
import re
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
plt.ion()

matplotlib.rc('xtick', labelsize=30)
matplotlib.rc('ytick', labelsize=30)


class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure(figsize=(16,16))
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        plt.figure(figsize=(16, 16))
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.xlabel('Epoch', fontsize=30)
        plt.ylabel('MSE', fontsize=30)
        plt.legend(prop={'size': 30})
        plt.grid()
        name = 'cv_' + str(epoch) + '.pdf'
        figure_path = os.path.join('Figures', name)
        plt.savefig(figure_path)
        plt.show(block=False)
        plt.draw()
        
        
class DeepLearningModel:
    def __init__(self, model_name, model_dir, model_subDir, input_dim,
                 output_dim, optimizer, metrics=None, loss="mse",
                 scale_pred=False, scale_target=True, add_earlyStopping = False,
                 audio_model=False, plot_loss=True, weights_path=None,
                 saved_weights=True, load_weights=False,
                 neuron_parameters={}, layers=5, initialization='he_normal',
                 level_dropout=.25, problem='regression'):

        self.logger = gv.logger
        self.model_name = model_name
        self.model_dir = model_dir
        self.model_subDir = os.path.join(self.model_dir,
                                         model_subDir)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.optimizer = optimizer
        self.metrics = metrics
        self.loss = loss
        self.scale_pred = scale_pred
        self.scale_target = scale_target
        self.add_earlyStopping = add_earlyStopping
        self.audio_model = audio_model
        if plot_loss:
            self.plot_loss = PlotLosses()
        else:
            self.plot_loss = None
        
        self.weights_path = weights_path
        self.saved_weights = saved_weights
        self.load_weights = load_weights
        self.neuron_parameters = neuron_parameters
        self.layers = layers
        self.initialization = initialization
        self.level_dropout = level_dropout
        self.problem=problem
        self.saved_model_path = os.path.join(self.model_subDir, self.model_name)

        if self.weights_path is not None:
            weight_name = self.weights_path
            self.weights_path = os.path.join(self.saved_model_path, weight_name)

        # Build Directory
        self.init_directories()

        self.model = None

    def init_directories(self):
        try:
            # Create Root Directory
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            # Create sub-directories
            if not os.path.exists(self.model_subDir):
                os.makedirs(self.model_subDir)

            # Model Directory
            if not os.path.exists(self.saved_model_path):
                os.makedirs(self.saved_model_path)
        except Exception as e:
            self.logger.error(e)
            self.logger.error('Unable to create model directory')

    def build_model(self):
        ok = True
        try:
            self.model = self.create_model()
        except Exception as e:
            self.logger.error(e)
            ok = False
            self.model = None
        return ok
    
    def create_model(self):
        
        model = Sequential()
        
        for i in range(self.layers):
            if i == 0:
                # Input Layer
                n_layer = 'initial'
                n_hidden_neurons = self.get_hidden_neurons(self.input_dim,
                                                           self.output_dim,
                                                           n_layer,
                                                           self.neuron_parameters)
                model.add(Dense(n_hidden_neurons, input_dim=self.input_dim,
                                activation='relu',
                                kernel_initializer= self.initialization))
                model.add(Dropout(self.level_dropout))
            elif i > 0 and i < self.layers - 2:
                # Hidden Layers
                n_layer = 'intermediate'
                n_hidden_neurons = self.get_hidden_neurons(self.input_dim,
                                                           self.output_dim,
                                                           n_layer,
                                                           self.neuron_parameters)
                model.add(Dense(n_hidden_neurons, activation='relu',
                                kernel_initializer= self.initialization)) 
                model.add(Dropout(self.level_dropout))
                
            elif i == self.layers-2:
                # Previous output Layer
                n_layer = 'final'
                n_hidden_neurons = self.get_hidden_neurons(self.input_dim,
                                                           self.output_dim,
                                                           n_layer,
                                                           self.neuron_parameters)
                model.add(Dense(n_hidden_neurons, activation='relu',
                                kernel_initializer= self.initialization))
                
            elif i == self.layers-1:
                # OutputLayer
                if self.scale_target and self.problem=='regression':
                    activate_funct = 'sigmoid'
                elif not self.scale_target and self.problem=='regression':
                    activate_funct = 'linear'
                elif self.problem=='classification' and self.scale_target: # Classification
                    activate_funct = 'softmax'
                else:
                    activate_funct=None
                model.add(Dense(self.output_dim, activation=activate_funct))
        return model

    def compile_model(self):
        ok=True
        try:
            if self.load_weights and self.weights_path is not None:
                self.model.load_weights(self.weights_path)

            # Add classification metrics
            if self.problem == 'classification':
                self.metrics +=[f1_m, precision_m, recall_m]

            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
            self.model.summary()
            self.logger.info('Model Compiled!')
        except Exception as e:
            self.logger.error(e)
            self.logger.error('Unable to compile the model')
            ok=False
        return ok

    def get_hidden_neurons(self, input_size, output_size=1, n_layer='initial',
                           neuron_parameters=None):
        
        alpha = neuron_parameters['alpha'] # 5
        beta = neuron_parameters['beta']  # 6/3
        gamma = neuron_parameters['gamma'] # 3

        if n_layer == 'initial':
            n_hidden_neurons = int(round(alpha*input_size)) # 2*input_size
        elif n_layer == 'intermediate':
            n_hidden_neurons = int(round((beta*input_size))) # beta = 2/3
        else:
            n_hidden_neurons = int(round(gamma*input_size)) # 1/2
        return n_hidden_neurons

    def train_model(self, X_train, y_train, X_test, y_test, epochs, batch_size,
                    monitor_early='val_mean_absolute_error', mode='min',
                    monitor_checkout='val_loss'):
        
        callback_elements = []
        add_callback = False
        # Training model with train data. Fixed random seed:
        if self.add_earlyStopping:
            n_patience = np.round(epochs*0.05)
            
            if self.problem == 'classification':
                mode_early = 'auto'
            else:
                mode_early = mode
            earlystop = EarlyStopping(monitor=monitor_early,
                                      patience=n_patience,
                                      verbose=1, mode=mode_early)
            callback_elements.append(earlystop)
            add_callback=True
            
        if self.plot_loss is not None:
            callback_elements.append(self.plot_loss)
            add_callback = True
        
        if self.saved_weights and self.weights_path is not None:
            
            checkpoint = ModelCheckpoint(self.weights_path,
                                         monitor=monitor_checkout,
                                         verbose=1,
                                         save_best_only=True,
                                         mode=mode)
            callback_elements.append(checkpoint)
            add_callback = True

        # Add TensorBoard
        tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=10, write_graph=True, write_images=True)

        callback_elements.append(tbCallBack)

        # Add Categorical
        if self.problem == 'classification':
            # Compute weights
            class_weights = class_weight.compute_class_weight('balanced',
                                                              np.unique(y_train),
                                                              y_train)
            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)
        else:
            class_weights = None
            
        # Add Callback
        if add_callback:
            history = self.model.fit(X_train, y_train, epochs=epochs, 
                                     batch_size=batch_size, verbose=2,
                                     validation_data=(X_test, y_test),
                                     class_weight=class_weights,
                                     callbacks=callback_elements)
        else:
            history = self.model.fit(X_train, y_train, epochs=epochs, 
                                     batch_size=batch_size, verbose=2,
                                     validation_data=(X_test, y_test),
                                     class_weight=class_weights)
            
        # Plot Accuracy
        """plt.figure(figsize=(16,16))
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.legend(['acc', 'val_acc'], loc='upper left',
                   prop={'size': 30})
        plt.xlabel('Epoch', fontsize=30)
        plt.ylabel('Accuracy', fontsize=30)
        plt.grid()
        name = 'cv_class' + '.pdf'
        figure_path = os.path.join('Figures', name)
        plt.savefig(figure_path)"""
        return history

    def validate_model(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

    def save_history(self, model_history, k):
        ok = True
        try:
            model_history_name = 'model_history_' + str(k + 1) + '.json'
            model_history_path = os.path.join(self.saved_model_path, model_history_name)
            with open(model_history_path, 'w') as f:
                json.dump(model_history.history, f)
        except Exception as e:
            self.logger.error(e)
            ok = False
        return ok

    def save_model(self):
        ok = True
        model_path = os.path.join(self.saved_model_path, 'model.json')
        try:
            model_json = self.model.to_json()
            with open(model_path, "w") as json_file:
                json_file.write(model_json)
            self.model.save_weights(self.weights_path)
            self.logger.info("Saved trained model at %s", model_path)
        except Exception as e:
            self.logger.error(e)
            self.logger.error("Unable to save model at %s", model_path)
            ok = False
        return ok

    def load_model(self):
        model = None
        try:
            model = model_from_json(self.saved_model_path)
            model.load_weights(self.weights_path)
        except Exception as e:
            self.logger.error('Unable to load Neural Network model')
            self.logger.error(e)

        return model

    def train_autoencoder(self, x_train, epochs=50, compress='auto'):
        encoder = None
        code_size = 1
        try:
            input_size = x_train.shape[1]
            hidden_size = int(input_size / 2)
            hidden_size_2 = int(input_size / 3)

            # Compress Level
            self.logger.info('Compress level in the Autoencoder: %s', compress)
            if compress == 'auto':
                code_size = int(input_size/5)
            elif compress == 'high':
                code_size = int(input_size/7)
            elif compress == 'small':
                code_size = int(input_size/4)

            if isinstance(x_train, pd.DataFrame):
                x_train = np.array(x_train)

            m = Sequential()
            m.add(Dense(hidden_size, activation='relu', input_shape=(input_size,)))
            m.add(Dense(hidden_size_2, activation='relu'))
            m.add(Dense(code_size, activation='relu', name='compressed'))
            m.add(Dense(hidden_size_2, activation='relu'))
            m.add(Dense(hidden_size, activation='relu'))
            m.add(Dense(input_size, activation='sigmoid'))

            # Compile Model
            m.compile(optimizer='adam', loss='mean_squared_error')
            earlystop = EarlyStopping(monitor='loss',
                                      patience=2,
                                      verbose=2,
                                      mode='min')
            m.fit(x_train, x_train, epochs=epochs, callbacks=[earlystop])

            # Compressed Feature Vector
            encoder = Model(m.input, m.get_layer('compressed').output)

            # Save model
            model_path = os.path.join(self.saved_model_path, 'autoencoder.json')
            weights_path = os.path.join(self.saved_model_path, 'autoencoder' + '_weights.h5')
            autoencoder_json = m.to_json()

            with open(model_path, "w") as json_file:
                json_file.write(autoencoder_json)

            self.logger.info("Saved trained model at %s", model_path)
            # save weights
            m.save_weights(weights_path)
            self.logger.info("Saved weights at %s", weights_path)
        except Exception as e:
            self.logger.error(e)

        return encoder

    def get_model_summary(self, history=None, task='regression', k_fold=5):
        output = {}
        try:
            # Regression
            if task == 'regression':
                for key in list(history.keys()):
                    output['avg_' + key] = np.mean(history[key])
                    self.logger.info('Average %s: %s', str(key), np.mean(history[key]))
            # Classification
            else:
                for key in list(history.keys()):
                    # No Number in the last position
                    if not re.search('\d+', key.split('_')[-1]) and '_m' in key:
                        temp = [history[key]]
                        all_keys = [key + '_' + str(i) for i in range(1, k_fold)]
                        all_values = [history[k] for k in all_keys]
                        all_values += temp
                        output[key] = np.mean(all_values)
                    elif not re.search('\d+', key) and '_m' not in key:
                        output[key] = history[key]


            # Save Results
            results_path = os.path.join(self.saved_model_path, 'model_summary.json')
            jsonobj = json.dumps(output)
            with open(results_path, "w") as json_file:
                json_file.write(jsonobj)
        except Exception as e:
            self.logger.error(e)
            output = {''}

        return output

    def model_evaluation_classifier(self, x_test, y_test):
        metrics = {}
        try:
            # 1) Accuracy
            # evaluate the model
            loss, accuracy, f1_score, precision, recall = self.model.evaluate(x_test, y_test,
                                                                              verbose=0)
            metrics['accuracy'] = accuracy
            metrics['f1_score'] = f1_score
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['loss'] = loss
        except Exception as e:
            print(e)
            metrics = {}
        return metrics