from keras.models import model_from_json
from helper import global_variables as gv
from models.classification_metrics_dnn import recall_m, precision_m, f1_m

import os
import pandas as pd
import numpy as np

gv.init()

dm = gv.data_manager

# Get a track
df_tracks = pd.read_csv(os.path.join('input', 'df_tracks.csv'), index_col=0)
track_ids = pd.read_csv(os.path.join('input', 'tracks_id.csv'), index_col=0)
x_test_reg = pd.read_csv(os.path.join('input', 'x_test_regression.csv'), index_col=0)
y_test_reg = pd.read_csv(os.path.join('input', 'y_test_regression.csv'),header=None, index_col=0)
y_test_cls = pd.read_csv(os.path.join('input', 'y_test_classification.csv'), header=None, index_col=0)

# Load musicPopNet both regression and classifier
with open(os.path.join('saved_models', 'feature_compressed_auto', 'model_1_A','model.json'), 'r') as f:
    musicPopNet = model_from_json(f.read())

# Regression
musicPopNet.load_weights(os.path.join('saved_models', 'feature_compressed_auto', 'model_1_A','weights.h5'))
musicPopNet.compile(loss='mse',optimizer='adadelta', metrics=['mae'])
x_tes_t = x_test_reg.drop(['track_id'], axis=1, inplace=False)
y_pred = musicPopNet.predict(np.array(x_tes_t))
mae_reg = np.abs(y_pred - y_test_reg)
mae_reg['n_row'] = [i for i in range(len(mae_reg))]
# Get the ids of the 3 best classified and the worst 3
data_Reg = mae_reg.sort_values(by=[1])

best_values = data_Reg[0:3]
worst_values = data_Reg[8500:8503]

best_ids = [track_ids.loc[tr_id, 'track_id'] for tr_id in list(best_values.index)] 
worst_ids = [track_ids.loc[tr_id, 'track_id'] for tr_id in list(worst_values.index)] 

for value in list(best_values.n_row):
    print(y_pred[value])
    print(np.array(y_test_reg)[value])
    
for value in list(worst_values.n_row):
    print(y_pred[value])
    print(np.array(y_test_reg)[value])

# Classification
with open(os.path.join('saved_models', 'class_feature_compressed_auto', 'model_1_A','model.json'), 'r') as f:
    musicPopNetClass = model_from_json(f.read())

musicPopNetClass.load_weights(os.path.join('saved_models', 'class_feature_compressed_auto', 'model_1_A','weights.h5'))

metrics=['accuracy']
metrics += [f1_m, precision_m, recall_m]
musicPopNetClass.compile(loss='categorical_crossentropy',optimizer='adadelta', metrics=metrics)
y_pred_cls = musicPopNetClass.predict(np.array(x_tes_t))
y_classes = y_pred_cls.argmax(axis=-1)
y_test_cls['n_row'] = [i for i in range(len(y_test_cls))]



worst_values_clas = [36591, 34127, 27336]
best_values_clas = [89468, 87303, 13163]
best_ids = [track_ids.loc[tr_id, 'track_id'] for tr_id in best_values_clas] 
worst_ids = [track_ids.loc[tr_id, 'track_id'] for tr_id in worst_values_clas] 
best_values_rows = [1,191, 1166]
worst_values_clas = [27, 1190, 1382]

for value in best_values_rows:
    print(y_pred[value])
    print(np.array(y_test_reg)[value])
    
for value in worst_values_clas:
    print(y_pred[value])
    print(np.array(y_test_reg))
