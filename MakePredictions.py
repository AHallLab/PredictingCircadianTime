from pickle import load
import pandas as pd
import numpy as np
import math
import tensorflow as tf
from sklearn.preprocessing import normalize

x_data = pd.read_csv('Data\\X_train_raw.txt').T
x_data.columns = x_data.iloc[0]
genes = pd.read_csv('CircadianGenes.csv', header=None)
x_data = x_data.loc[:, genes.values.ravel()]
x_data = x_data.iloc[1:, :]

def predict_time(x_data):
    scaler = load(open('scaler.pkl', 'rb'))
    x_data_scaled = scaler.transform(x_data)
    all_preds = []
    for i in range(12):
        model = tf.keras.models.load_model('CircadianArabidopsisModel{}.hdf5'.format(i), compile=False)
        all_preds.append(normalize(model.predict(x_data_scaled)))
    time_preds = np.mean(all_preds, axis=0)
    time_angles = []
    for i in range(time_preds.shape[0]):
        time_angles.append(math.atan2(time_preds[i, 0], time_preds[i, 1]) / math.pi * 12)
    for j in range(len(time_angles)):
        if time_angles[j] < 0:
            time_angles[j] = time_angles[j] + 24

    preds = np.vstack(time_angles)
    return all_preds, preds

time_preds = predict_time(x_data)
