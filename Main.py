import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal as mvn
import seaborn as sn
import math
import gc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, normalize
import FourierClock
from scipy.stats import ks_2samp
from functools import reduce
import random
import os
from numpy.linalg import norm
import subprocess
from copulas.multivariate import GaussianMultivariate



val_errors1 = []
test_errors1 = []

N_GENES = 30
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

df = pd.read_csv('Data\\X_train_raw.csv').T
df_valid = pd.read_csv('Data\\X_valid_raw.csv').T
df_test = pd.concat((pd.read_csv('Data\\X_test_raw_A.txt').T, pd.read_csv('Data\\X_test_raw_B.txt').T)).iloc[[0, 1, 2, 4, 5], :]
rach_clusters = pd.read_csv('Data\\X_train_clusters.csv')
Y_data = df.iloc[1:, -1].astype('float64')
Y_copy = Y_data
Y_valid_data = df_valid.iloc[1:, -1].astype('float64')
Y_valid_copy = Y_valid_data

common_IDs = reduce(np.intersect1d, (df.iloc[0, :-1].values, df_valid.iloc[0, :-1].values, df_test.iloc[0, :].values))

idx = np.where(df.iloc[0, :].isin(common_IDs))[0]
df = df.iloc[:, idx]
idx_valid = np.where(df_valid.iloc[0, :].isin(common_IDs))[0]
df_valid = df_valid.iloc[:, idx_valid]
idx_test = np.where(df_test.iloc[0, :].isin(common_IDs))[0]
df_test = df_test.iloc[:, idx_test]

X_data = df.iloc[1:, :].astype('float64')
X_ID = df.iloc[0, :]
X_valid_data = df_valid.iloc[1:, :].astype('float64')
X_valid_ID = df_valid.iloc[0, :]
X_test_data = df_test.iloc[1:, :].astype('float64')
X_test_ID = df_test.iloc[0, :]

X_ID1 = np.argsort(X_ID)
X_ID = X_ID.iloc[X_ID1]
X_data = X_data.iloc[:, X_ID1]
X_data.columns = X_ID
X_ID1 = np.argsort(X_valid_ID)
X_valid_ID = X_valid_ID.iloc[X_ID1]
X_valid_data = X_valid_data.iloc[:, X_ID1]
X_valid_data.columns = X_valid_ID
X_ID1 = np.argsort(X_test_ID)
X_test_ID = X_test_ID.iloc[X_ID1]
X_test_data = X_test_data.iloc[:, X_ID1]
X_test_data.columns = X_test_ID

# Variance threshold
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold()
selector.fit(X_data)
var_idx = selector.variances_ > 5
X_data = X_data.iloc[:, var_idx]
X_ID = X_ID.iloc[var_idx]
X_valid_data = X_valid_data.iloc[:, var_idx]
X_valid_ID = X_valid_ID.iloc[var_idx]
X_test_data = X_test_data.iloc[:, var_idx]
X_test_ID = X_test_ID.iloc[var_idx]

X_data.reset_index(inplace=True, drop=True)
X_valid_data.reset_index(inplace=True, drop=True)
X_test_data.reset_index(inplace=True, drop=True)

X_ID.reset_index(inplace=True, drop=True)
X_valid_ID.reset_index(inplace=True, drop=True)
X_test_ID.reset_index(inplace=True, drop=True)

del df
gc.collect()

n_folds = Y_data.shape[0]
folds = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

y_cos = -np.cos((2 * np.pi * Y_data.astype('float64') / 24)+(np.pi/2))
y_sin = np.sin((2 * np.pi * Y_data.astype('float64') / 24)+(np.pi/2))

Y_valid_cos = -np.cos((2 * np.pi * Y_valid_data.astype('float64') / 24)+(np.pi/2))
Y_valid_sin = np.sin((2 * np.pi * Y_valid_data.astype('float64') / 24)+(np.pi/2))

def cyclical_loss(y_true, y_pred):
    error = 0
    for i in range(y_pred.shape[0]):
        error += np.arccos((y_true[i, :] @ y_pred[i, :]) / (norm(y_true[i, :]) * norm(y_pred[i, :])))
    return error

def custom_loss(y_true, y_pred):
    return tf.reduce_mean((tf.math.acos(tf.matmul(y_true, tf.transpose(y_pred)) / ((tf.norm(y_true) * tf.norm(y_pred)) + tf.keras.backend.epsilon()))**2))

adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)


def larger_model():
    # create model
    model = Sequential()
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, kernel_initializer='normal'))
    # Compile model
    model.compile(loss=custom_loss, optimizer=adam)
    return model

Y_data = np.concatenate((y_cos.values.reshape(-1, 1), y_sin.values.reshape(-1, 1)), axis=1)
Y_valid_data = np.concatenate((Y_valid_cos.values.reshape(-1, 1), Y_valid_sin.values.reshape(-1, 1)), axis=1)

error = 0  # Initialise error
all_preds = np.zeros((Y_data.shape[0], 2))  # Create empty array
all_valid_preds = np.zeros((Y_valid_data.shape[0], 2))  # Create empty array
early_stop = EarlyStopping(patience=100, restore_best_weights=True, monitor='val_loss', mode='min')

X_data_times = X_data.T
Y_times = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44])
scaler = StandardScaler()
X_data_times_idx = X_data_times.index
X_data_times = (scaler.fit_transform(X_data_times.T)).T
X_data_times = pd.DataFrame(data=X_data_times, index=X_data_times_idx)
X_data_times = pd.concat((pd.DataFrame(Y_times.reshape(1, 12)), X_data_times), axis=0)
X_data_times.to_csv('Data\\X_train_times.csv', header=None)

subprocess.call(['C:\\Program Files\\R\\R-4.0.3\\bin\\Rscript', 'metacycle_scores.R'], shell=False)

arser_scores = pd.read_csv('MetaScores\\ARSresult_X_train_times.csv')
jtk_scores = pd.read_csv('MetaScores\\JTKresult_X_train_times.csv')

auto_indices, auto_clock_genes, auto_scores = FourierClock.get_autocorrelated_genes(X_data, X_ID)
auto_scores = np.abs(np.array(auto_scores))

cross_indices, cross_clock_genes, cross_scores = FourierClock.cross_corr(X_data, Y_copy, X_ID)
cross_scores = np.abs(np.array(cross_scores))

scores = np.concatenate((auto_scores.reshape(-1, 1), cross_scores.reshape(-1, 1),
                         arser_scores['fdr_BH'].values.reshape(-1, 1), jtk_scores['ADJ.P'].values.reshape(-1, 1)),
                        axis=1)

scores[:, 2:] = 1-scores[:, 2:]

num_resamples = 1000 # Change to 50,000/100,000

gcopula = GaussianMultivariate()
gcopula.fit(scores)
random_sample = gcopula.sample(num_resamples)
sample_scores = pd.DataFrame(random_sample)
mean = np.mean(sample_scores.values, axis=0)
covariance = np.cov(sample_scores.T)
dist = mvn(mean=mean, cov=covariance, allow_singular=True)

gene_scores = []
for i in range(scores.shape[0]):
    gene_scores.append(dist.cdf(x=scores[i, :]))

gene_scores = np.array(gene_scores)
gene_scores = np.concatenate((arser_scores['CycID'].values.reshape(-1, 1), gene_scores.reshape(-1, 1)), axis=1)


gene_scores = gene_scores[gene_scores[:, 1].argsort()[::-1]]
selected_genes = gene_scores[:N_GENES*3, 0]

idx = np.where(X_ID.isin(selected_genes))[0]
selected_scores = gene_scores[idx]
X_data = X_data.iloc[:, idx]
idx_valid = np.where(X_valid_ID.isin(selected_genes))[0]
X_valid_data = X_valid_data.iloc[:, idx_valid]
idx_test = np.where(X_test_ID.isin(selected_genes))[0]
X_test_data = X_test_data.iloc[:, idx_test]

X_ID = X_ID.iloc[idx]
X_valid_ID = X_valid_ID.iloc[idx_valid]
X_test_ID = X_test_ID.iloc[idx_test]

scores = []
pvalues = []

for i in range(X_data.shape[1]):
    l = ks_2samp(X_data.iloc[:, i], X_valid_data.iloc[:, i])
    scores.append(i)
    pvalues.append(l.pvalue)

pvalues_idx = np.argsort(pvalues)
scores = pvalues_idx[(pvalues_idx.shape[0]-2*N_GENES):]

similar_genes = selected_genes[scores]
X_data = X_data.iloc[:, scores]
selected_scores = selected_scores[scores]
X_ID = X_ID.iloc[scores]
X_valid_data = X_valid_data.iloc[:, scores]
X_test_data = X_test_data.iloc[:, scores]

Y_copy_res = np.array([0, 4, 8, 12, 16, 20, 0, 4, 8, 12, 16, 20])

X_ID2 = X_data.columns.values

scaler = MinMaxScaler()
scaler.fit(X_data)
X_data = scaler.transform(X_data)
X_valid_data = scaler.transform(X_valid_data)
X_test_data = scaler.transform(X_test_data)

X_data = pd.DataFrame(data=X_data, columns=X_ID2)
X_valid_data = pd.DataFrame(data=X_valid_data, columns=X_ID2)
X_test_data = pd.DataFrame(data=X_test_data, columns=X_ID2)

column_max = np.max(X_valid_data.values, axis=0)
column_min = np.min(X_valid_data.values, axis=0)
column_idx = column_max < 1.4
column_idx1 = column_min > -0.2
column_idx = np.logical_and(column_idx, column_idx1)
X_data = X_data.iloc[:, column_idx]
selected_scores = selected_scores[column_idx]
X_ID = X_ID.iloc[column_idx]
X_valid_data = X_valid_data.iloc[:, column_idx]
X_test_data = X_test_data.iloc[:, column_idx]

P = pd.DataFrame(data=selected_scores, columns=['transcript', 'score'])
L = pd.merge(P, rach_clusters, how='left', left_on='transcript', right_on='transcript')

L.fillna(value='None', inplace=True)
print(L['moduleColor'].value_counts())

colours = L['moduleColor'].unique()

IDs = []

for i in range(colours.shape[0]):
    colour = colours[i]
    genes = L.loc[L['moduleColor'] == colour]
    genes.sort_values(inplace=True, by='score')
    genes = genes.iloc[-int(N_GENES/9):]
    IDs.append(genes['transcript'])

all_IDs = pd.concat((IDs))
X_IDs = np.array(all_IDs.index)

X_data = X_data.iloc[:, X_IDs]
X_valid_data = X_valid_data.iloc[:, X_IDs]
X_test_data = X_test_data.iloc[:, X_IDs]

M = L.iloc[X_IDs]

valid_preds = []
test_preds = []

X_data = X_data.values
X_valid_data = X_valid_data.values
X_test_data = X_test_data.values

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_data, Y_data)):
    X_train, Y_train = X_data[train_idx], Y_data[train_idx]  # Define training data for this iteration
    X_valid, Y_valid = X_data[valid_idx], Y_data[valid_idx]

    model = larger_model()
    model.fit(X_train.astype('float64'), Y_train.astype('float64'), validation_data=(X_valid.astype('float64'), Y_valid.astype('float64')),
              batch_size=2, epochs=5000, callbacks=[early_stop])  # Fit the model on the training data

    preds = normalize(model.predict(X_valid))  # Predict on the validation data
    all_preds[valid_idx] = normalize(model.predict(X_valid))
    all_valid_preds += (normalize(model.predict(X_valid_data)) / n_folds)
    valid_preds.append(normalize(model.predict(X_valid_data)))
    test_preds.append(normalize(model.predict(X_test_data)))
    error += cyclical_loss(Y_valid.astype('float64'), preds.astype('float64'))  # Evaluate the predictions
    print(cyclical_loss(Y_valid.astype('float64'), preds.astype('float64')) / Y_valid.shape[0])

angles = []
for i in range(all_preds.shape[0]):
    angles.append(math.atan2(all_preds[i, 0], all_preds[i, 1]) / math.pi * 12)

for j in range(len(angles)):
    if angles[j] < 0:
        angles[j] = angles[j] + 24

ax = sn.scatterplot(Y_data[:, 0], Y_data[:, 1])
ax = sn.scatterplot(all_preds[:, 0], all_preds[:, 1])
plt.show()
angles_arr = np.vstack(angles)
hour_pred = angles_arr

plt.figure(dpi=500)
ax = sn.lineplot(np.arange(Y_copy.shape[0]), Y_copy)
ax = sn.lineplot(np.arange(Y_copy.shape[0]), angles_arr.ravel())
plt.show()


angles = []
for i in range(all_preds.shape[0]):
    angles.append(math.atan2(all_preds[i, 0], all_preds[i, 1]) / math.pi * 12)

for j in range(len(angles)):
    if angles[j] < 0:
        angles[j] = angles[j] + 24


valid_angles = []
valid_preds = np.mean(valid_preds, axis=0)
for i in range(valid_preds.shape[0]):
    valid_angles.append(math.atan2(valid_preds[i, 0], valid_preds[i, 1]) / math.pi * 12)

for j in range(len(valid_angles)):
    if valid_angles[j] < 0:
        valid_angles[j] = valid_angles[j] + 24
valid_preds = normalize(valid_preds)
ax = sn.scatterplot(Y_valid_data[:, 0], Y_valid_data[:, 1])
ax = sn.scatterplot(valid_preds[:, 0], valid_preds[:, 1])
plt.show()
angles_arr_valid = np.vstack(valid_angles)
hour_pred_valid = angles_arr_valid


plt.figure(dpi=500)
ax = sn.lineplot(np.arange(Y_valid_copy.shape[0]), Y_valid_copy)
ax = sn.lineplot(np.arange(Y_valid_copy.shape[0]), angles_arr_valid.ravel())
plt.show()

# print("Average error = {}".format(cyclical_loss(Y_data.astype('float64'), all_preds.astype('float64')) / Y_data.shape[0]))
print("Average training error = {} minutes".format(60 * 12 * cyclical_loss(Y_data.astype('float64'), all_preds.astype('float64')) / (Y_data.shape[0] * np.pi)))

# print("Average error = {}".format(cyclical_loss(Y_valid_data.astype('float64'), all_valid_preds.astype('float64')) / Y_valid_data.shape[0]))
print("Average validation error = {} minutes".format(60 * 12 * cyclical_loss(Y_valid_data.astype('float64'), valid_preds.astype('float64')) / (Y_valid_data.shape[0] * np.pi)))

Y_copy1 = np.array([2, 5, 8, 11, 14, 17, 20, 23, 2, 5, 8, 11, 14, 17, 20, 23])

test_angles = []
test_preds_copy = test_preds
test_preds = np.mean(test_preds, axis=0)
for j in range(len(test_preds_copy)):
    for i in range(test_preds.shape[0]):
        test_preds_copy[j][i, 0] = math.atan2(test_preds_copy[j][i, 0], test_preds_copy[j][i, 1]) / math.pi * 12
        if test_preds_copy[j][i, 0] < 0:
            test_preds_copy[j][i, 0] += 24
    test_preds_copy[j] = np.delete(test_preds_copy[j], 1, 1)

for i in range(test_preds.shape[0]):
    test_angles.append(math.atan2(test_preds[i, 0], test_preds[i, 1]) / math.pi * 12)
for j in range(len(test_angles)):
    if test_angles[j] < 0:
        test_angles[j] = test_angles[j] + 24
test_preds = normalize(test_preds)
angles_arr_test = np.vstack(test_angles)
hour_pred_test = angles_arr_test
Y_test = np.array([12, 0, 12, 0])

Y_test_cos = -np.cos((2 * np.pi * Y_test.astype('float64') / 24) + (np.pi / 2))
Y_test_sin = np.sin((2 * np.pi * Y_test.astype('float64') / 24) + (np.pi / 2))
Y_test_ang = np.concatenate((Y_test_cos.reshape(-1, 1), Y_test_sin.reshape(-1, 1)), axis=1)
print("Average test error = {} minutes".format(60 * 12 * cyclical_loss(Y_test_ang.astype('float64'), test_preds.astype('float64')) / (Y_test_ang.shape[0] * np.pi)))
val_errors1.append(60 * 12 * cyclical_loss(Y_valid_data.astype('float64'), all_valid_preds.astype('float64')) / (Y_valid_data.shape[0] * np.pi))
test_errors1.append(60 * 12 * cyclical_loss(Y_test_ang.astype('float64'), test_preds.astype('float64')) / (Y_test_ang.shape[0] * np.pi))
