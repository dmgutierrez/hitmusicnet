import matplotlib.pyplot as plt
import json
import os
import numpy as np


# Load history




# Classification
history = []

for i in range(1,6):
    name = 'model_history_' + str(i) + '.json'
    with open(os.path.join('saved_models', 'class_feature_compressed_auto', 'model_1_A',name), 'r') as f:
        history.append(json.load(f))


# Get mean Val_loss/Loss
"""min_epoch = 16

mean_val_loss = np.hstack((np.array(0.479876), np.mean([vl['val_loss'][0:min_epoch] for vl in history],
                               axis=0), np.array([0.4191876, 0.41567])))
mean_loss = np.hstack((np.array(0.479876), np.mean([vl['loss'][0:min_epoch] for vl in history],
                           axis=0), np.array([0.4219876, 0.42287])))

# Get Mean Val_acc/acc
mean_val_acc = np.hstack((np.array(0.769876), np.mean([vl['val_acc'][0:min_epoch] for vl in history],
                          axis=0), np.array([0.8310, 0.8305])))

mean_acc = np.hstack((np.array(0.759876), np.mean([vl['acc'][0:min_epoch] for vl in history],
                      axis=0), np.array([0.8305, 0.8301])))



epochs = [2*i for i in range(min_epoch+3)]"""


figure_path = 'Figures/accuracy.pdf'
plt.figure(figsize=(16, 16))
plt.plot(history[0]['acc'])
plt.plot(history[0]['val_acc'])
plt.legend(['acc', 'val_acc'], loc='lower right',
           prop={'size': 30})
plt.xlabel('Epoch', fontsize=30)
plt.ylabel('Accuracy', fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.grid()
plt.savefig(figure_path, bbox_inches='tight')
plt.show()


# ----------------------------------
figure_path = 'Figures/loss.pdf'
plt.figure(figsize=(16, 16))
plt.plot(history[0]['loss'])
plt.plot(history[0]['val_loss'])
plt.legend(['loss', 'val_loss'], loc='upper right',
           prop={'size': 30})
plt.xlabel('Epoch', fontsize=30)
plt.ylabel('Categorical Cross-Entropy', fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

plt.grid()
plt.savefig(figure_path,bbox_inches='tight')
plt.show()
