import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import math
import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import normalize
import random
from numpy.linalg import norm

def reset_seeds(reset_graph_with_backend=None):
    if reset_graph_with_backend is not None:
        K = reset_graph_with_backend
        K.clear_session()
        tf.compat.v1.reset_default_graph()
    np.random.seed(1)
    random.seed(2)
    tf.compat.v1.set_random_seed(3)


def SequentialFeatureSelectionCluster(max_genes, colours, L, X_data, Y_data, X_valid_data, Y_valid_data, X_test_data):
    n_genes = 0
    i = 0
    n_folds = Y_data.shape[0]
    SEED = 0
    folds = KFold(n_splits=n_folds, random_state=SEED)
    graph_res = []
    results1 = []

    def cyclical_loss(y_true, y_pred):
        error = 0
        for i in range(y_pred.shape[0]):
            error += np.arccos((y_true[i, :] @ y_pred[i, :]) / (norm(y_true[i, :]) * norm(y_pred[i, :])))
        return error

    def custom_loss(y_true, y_pred):
        return tf.reduce_mean((tf.math.acos(tf.matmul(y_true, tf.transpose(y_pred)) / (
                    (tf.norm(y_true) * tf.norm(y_pred)) + tf.keras.backend.epsilon())) ** 2))

    adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    early_stop = EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss', mode='min')

    def larger_model():
        # create model
        model = Sequential()
        model.add(Dense(32, kernel_initializer='normal', activation='relu'))
        model.add(Dense(128, kernel_initializer='normal', activation='relu'))
        model.add(Dense(512, kernel_initializer='normal', activation='relu'))
        model.add(Dense(2, kernel_initializer='normal'))
        # Compile model
        model.compile(loss=custom_loss, optimizer=adam)
        return model

    while n_genes < max_genes:
        n_genes += 1
        i %= colours.shape[0]
        colour = colours[i]
        genes = L.loc[L['moduleColor'] == colour]
        idx = genes.index.values
        result_iter = {'idx': [], 'train_error': [], 'val_error': [], 'test_error': []}

        for j in tqdm.tqdm(range(idx.shape[0])):
            idx1 = idx[j]
            if n_genes > 2:
                if idx1 in idx_perm:
                    result_iter['idx'].append(idx1)
                    result_iter['train_error'].append(999.99)
                    result_iter['val_error'].append(999.99)
                    result_iter['test_error'].append(999.99)
                    continue
            if counter == 1:
                idx1 = np.concatenate((np.array([idx1]), np.array(idx_perm).reshape(-1)))
            result_iter['idx'].append(idx1)
            X_d = X_data.iloc[:, idx1].values
            X_v = X_valid_data.iloc[:, idx1].values
            X_t = X_test_data.iloc[:, idx1].values

            valid_preds = []
            test_preds = []
            error = 0  # Initialise error
            all_preds = np.zeros((Y_data.shape[0], 2))  # Create empty array
            all_valid_preds = np.zeros((Y_valid_data.shape[0], 2))  # C

            for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_data, Y_data)):
                X_train, Y_train = X_d[train_idx], Y_data[train_idx]  # Define training data for this iteration
                X_valid, Y_valid = X_d[valid_idx], Y_data[valid_idx]
                if n_genes == 1:
                    X_train = X_train.reshape(X_train.shape[0], 1)
                    X_valid = X_valid.reshape(X_valid.shape[0], 1)
                    X_t = X_t.reshape(X_test_data.shape[0], 1)

                reset_seeds()
                model = larger_model()
                model.fit(X_train.astype('float64'), Y_train.astype('float64'), validation_data=(X_valid.astype('float64'), Y_valid.astype('float64')),
                          batch_size=3, epochs=100, callbacks=[early_stop], verbose=0)  # Fit the model on the training data
                preds = normalize(model.predict(X_valid))  # Predict on the validation data
                all_preds[valid_idx] = normalize(model.predict(X_valid))
                all_valid_preds += (normalize(model.predict(X_v)) / n_folds)
                valid_preds.append(normalize(model.predict(X_v)))
                test_preds.append(normalize(model.predict(X_t)))
                error += cyclical_loss(Y_valid.astype('float64'), preds.astype('float64'))  # Evaluate the predictions

            angles = []
            for k in range(all_preds.shape[0]):
                angles.append(math.atan2(all_preds[k, 0], all_preds[k, 1]) / math.pi * 12)

            for l in range(len(angles)):
                if angles[l] < 0:
                    angles[l] = angles[l] + 24

            angles = []
            for k in range(all_preds.shape[0]):
                angles.append(math.atan2(all_preds[k, 0], all_preds[k, 1]) / math.pi * 12)

            for l in range(len(angles)):
                if angles[l] < 0:
                    angles[l] = angles[l] + 24


            valid_angles = []
            valid_preds = np.mean(valid_preds, axis=0)
            for k in range(valid_preds.shape[0]):
                valid_angles.append(math.atan2(valid_preds[k, 0], valid_preds[k, 1]) / math.pi * 12)

            for m in range(len(valid_angles)):
                if valid_angles[m] < 0:
                    valid_angles[m] = valid_angles[m] + 24
            valid_preds = normalize(valid_preds)

            result_iter['train_error'].append(60 * 12 * cyclical_loss(Y_data.astype('float64'), all_preds.astype('float64')) / (Y_data.shape[0] * np.pi))
            result_iter['val_error'].append(60 * 12 * cyclical_loss(Y_valid_data.astype('float64'), valid_preds.astype('float64')) / (Y_valid_data.shape[0] * np.pi))

            test_angles = []
            test_preds_copy = test_preds
            test_preds = np.mean(test_preds, axis=0)
            for l in range(len(test_preds_copy)):
                for k in range(test_preds.shape[0]):
                    test_preds_copy[l][k, 0] = math.atan2(test_preds_copy[l][k, 0], test_preds_copy[l][k, 1]) / math.pi * 12
                    if test_preds_copy[l][k, 0] < 0:
                        test_preds_copy[l][k, 0] += 24
                test_preds_copy[l] = np.delete(test_preds_copy[l], 1, 1)

            for k in range(test_preds.shape[0]):
                test_angles.append(math.atan2(test_preds[k, 0], test_preds[k, 1]) / math.pi * 12)
            for m in range(len(test_angles)):
                if test_angles[m] < 0:
                    test_angles[m] = test_angles[m] + 24
            test_preds = normalize(test_preds)
            angles_arr_test = np.vstack(test_angles)
            hour_pred_test = angles_arr_test
            Y_test = np.array([12, 0, 12, 0])

            Y_test_cos = -np.cos((2 * np.pi * Y_test.astype('float64') / 24) + (np.pi / 2))
            Y_test_sin = np.sin((2 * np.pi * Y_test.astype('float64') / 24) + (np.pi / 2))
            Y_test_ang = np.concatenate((Y_test_cos.reshape(-1, 1), Y_test_sin.reshape(-1, 1)), axis=1)

            result_iter['test_error'].append(60 * 12 * cyclical_loss(Y_test_ang.astype('float64'), test_preds.astype('float64')) / (Y_test_ang.shape[0] * np.pi))
        i += 1
        counter = 1
        idx_perm = result_iter['idx'][result_iter['val_error'].index(min(result_iter['val_error']))]
        print(n_genes, idx[result_iter['val_error'].index(min(result_iter['val_error']))], result_iter['train_error'][result_iter['val_error'].index(min(result_iter['val_error']))], result_iter['val_error'][result_iter['val_error'].index(min(result_iter['val_error']))], result_iter['test_error'][result_iter['val_error'].index(min(result_iter['val_error']))])
        graph_res.append((n_genes, idx[result_iter['val_error'].index(min(result_iter['val_error']))], result_iter['train_error'][result_iter['val_error'].index(min(result_iter['val_error']))], result_iter['val_error'][result_iter['val_error'].index(min(result_iter['val_error']))], result_iter['test_error'][result_iter['val_error'].index(min(result_iter['val_error']))]))
        results1.append(result_iter)