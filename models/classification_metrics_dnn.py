from keras import backend as K


def recall_m(y_true, y_pred):
    try:
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
    except Exception as e:
        print(e)
    return recall


def precision_m(y_true, y_pred):
    try:
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
    except Exception as e:
        print(e)
    return precision


def f1_m(y_true, y_pred):
    try:
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
    except Exception as e:
        print(e)
        return None
    return 2*((precision*recall)/(precision+recall+K.epsilon()))