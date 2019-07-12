import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from pylab import rcParams

import tensorflow as tf
from keras import optimizers, Sequential
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score

from numpy.random import seed
seed(7)
from tensorflow import set_random_seed
set_random_seed(11)

from sklearn.model_selection import train_test_split

SEED = 123 #used to help randomly select the data points
DATA_SPLIT_PCT = 0.2

rcParams['figure.figsize'] = 8, 6
LABELS = ["Peace","War"]


# Shif Curve: This function will shift the binary labels in a dataframe.
# The curve shift will be with respect to the 1s. 
#    For example, if shift is -2, the following process
#    will happen: if row n is labeled as 1, then
#    - Make row (n+shift_by):(n+shift_by-1) = 1.
#    - Remove row n.
# i.e. the labels will be shifted up to 2 rows up.

sign = lambda x: (1, -1)[x < 0]
def curve_shift(df, shift_by):
    vector = df['y'].copy()
    for s in range(abs(shift_by)):
        tmp = vector.shift(sign(shift_by))
        tmp = tmp.fillna(0)
        vector += tmp
    labelcol = 'y'
    df.insert(loc=0, column=labelcol+'tmp', value=vector)
    df = df.drop(df[df[labelcol] == 1].index)
    df = df.drop(labelcol, axis=1)
    df = df.rename(columns={labelcol+'tmp': labelcol})
    df.loc[df[labelcol] > 0, labelcol] = 1
    return df

df86 = r.dat86
df89 = r.dat89

# df = curve_shift(df, shift_by=-1)

#####

# samples: This is simply the number of observations, or in other words, the number of data points.
# lookback: LSTM models are meant to look at the past. Meaning, at time t the LSTM will process data up to (t-lookback) to make a prediction.
# features: It is the number of features present in the input data.




input_X86 = df86.loc[:, df86.columns != 'y'].values  # converts the df to a numpy array
input_y86 = df86['y'].values

input_X89 = df89.loc[:, df89.columns != 'y'].values  # converts the df to a numpy array
input_y89 = df89['y'].values


n_features = input_X86.shape[1]  # number of features


# The input_X here is a 2-dimensional array of size samples x features. We want to be able to transform such a 2D array into a 3D array of size: samples x lookback x features. Refer to Figure 1 above for a visual understanding.

def temporalize(X, y, lookback, input_x, input_Y):
    X = []
    y = []
    for i in range(len(input_x)-lookback-1):
        t = []
        for j in range(1,lookback+1):
            t.append(input_x[[(i+j+1)], :])
        X.append(t)
        y.append(input_Y[i+lookback+1])
    return X, y

lookback = 4 # 3 Years of conflict data 
X86, y86 = temporalize(X = input_X86, y = input_y86, lookback = lookback, input_x = input_X86, input_Y = input_y86)
X89, y89 = temporalize(X = input_X89, y = input_y89, lookback = lookback, input_x = input_X89, input_Y = input_y89)


# Split into train, valid and test 

X_train, X_test, y_train, y_test = train_test_split(np.array(X86), np.array(y86), test_size=DATA_SPLIT_PCT, random_state=SEED)
X_valid = np.array(X89)
y_valid = np.array(y89)


# For training the autocoder, we will be using the X coming from only the negatively labeled data.
# Therefore, we separate the X corresponding to y = 0 

X_train_y0 = X_train[y_train==0]
X_train_y1 = X_train[y_train==1]

X_valid_y0 = X_valid[y_valid==0]
X_valid_y1 = X_valid[y_valid==1]



X_train = X_train.reshape(X_train.shape[0], lookback, n_features)
X_train_y0 = X_train_y0.reshape(X_train_y0.shape[0], lookback, n_features)
X_train_y1 = X_train_y1.reshape(X_train_y1.shape[0], lookback, n_features)

X_test = X_test.reshape(X_test.shape[0], lookback, n_features)

X_valid = X_valid.reshape(X_valid.shape[0], lookback, n_features)
X_valid_y0 = X_valid_y0.reshape(X_valid_y0.shape[0], lookback, n_features)
X_valid_y1 = X_valid_y1.reshape(X_valid_y1.shape[0], lookback, n_features)

X_train_y0.shape
# We will reshape the X's into the required 3D Dimension: (sample,lookback,features) 


def flatten(X):
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1]-1), :]
    return(flattened_X)

def scale(X, scaler):
    for i in range(X.shape[0]):
        X[i, :, :] = scaler.transform(X[i, :, :])
    return X

scaler = StandardScaler().fit(flatten(X_train_y0))
X_train_y0_scaled = scale(X_train_y0, scaler)
X_train_y1_scaled = scale(X_train_y1, scaler)
X_train_scaled = scale(X_train, scaler)

a = flatten(X_train_y0_scaled)
print('colwise mean', np.mean(a, axis=0).round(6))
print('colwise variance', np.var(a, axis=0))

X_valid_scaled = scale(X_valid, scaler)
X_valid_y0_scaled = scale(X_valid_y0, scaler)
X_test_scaled = scale(X_test, scaler)


timesteps =  X_train_y0_scaled.shape[1] # equal to the lookback
n_features =  X_train_y0_scaled.shape[2] 

epochs = 1000
batch = 64
lr = 0.00001

lstm_autoencoder = Sequential()
# Encoder
lstm_autoencoder.add(LSTM(30, activation='relu', input_shape=(timesteps, n_features), return_sequences=False))
lstm_autoencoder.add(RepeatVector(timesteps))
# Decoder
lstm_autoencoder.add(LSTM(30, activation='relu', return_sequences=True))
lstm_autoencoder.add(TimeDistributed(Dense(n_features)))

lstm_autoencoder.summary()

adam = optimizers.Adam(lr)
lstm_autoencoder.compile(loss='mse', optimizer=adam)

cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier.h5",
                               save_best_only=True,
                               verbose=0)

tb = TensorBoard(log_dir='./logs',
                histogram_freq=0,
                write_graph=True,
                write_images=True)

lstm_autoencoder_history = lstm_autoencoder.fit(X_train_y0_scaled, X_train_y0_scaled, 
                                                epochs=epochs, 
                                                batch_size=batch, 
                                                validation_data=(X_valid_y0_scaled, X_valid_y0_scaled),
                                                verbose=2).history

plt.plot(lstm_autoencoder_history['loss'], linewidth=2, label='Train')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()


valid_x_predictions = lstm_autoencoder.predict(X_valid_scaled)
mse = np.mean(np.power(flatten(X_valid_scaled) - flatten(valid_x_predictions), 2), axis=1)


error_df = pd.DataFrame({'Reconstruction_error': mse,
                        'True_class': y_valid.tolist()})


error_df

test_x_predictions = lstm_autoencoder.predict(X_test_scaled)
mse = np.mean(np.power(flatten(X_test_scaled) - flatten(test_x_predictions), 2), axis=1)

error_df = pd.DataFrame({'Reconstruction_error': mse,
                        'True_class': y_test.tolist()})
threshold_fixed = 0.026
groups = error_df.groupby('True_class')
fig, ax = plt.subplots()
pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]

conf_matrix = confusion_matrix(error_df.True_class, pred_y)

result = conf_matrix[1][1]/(conf_matrix[1][1]+conf_matrix[1][0])
print("Prediction Rate: %s" %(result))

















