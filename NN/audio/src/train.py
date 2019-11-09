
# https://towardsdatascience.com/getting-started-on-deep-learning-for-audio-data-667d9aa76a33
# https://www.tensorflow.org/lite/convert/python_api

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import KFold, cross_val_score, train_test_split

plt.style.use('fivethirtyeight')
print(os.listdir("../input"))

train = pd.read_json('../input/train.json')

# Train is a pandas dataframe of the format
#   >>> train.dtypes
#   audio_embedding                    object https://github.com/tensorflow/models/tree/master/research/audioset/vggish
#   end_time_seconds_youtube_clip       int64
#   is_turkey                           int64
#   start_time_seconds_youtube_clip     int64
#   vid_id                             object
#   dtype: object

print("Shape of train data : ", train.shape)

import keras
from keras import models
from keras import optimizers
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import BatchNormalization, Bidirectional, LSTM, TimeDistributed
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv1D, MaxPooling1D,GlobalMaxPooling1D
from keras.utils import np_utils

train_data, val_data = train_test_split(train, test_size=0.33, random_state = 42) # Split into training set and validation set

xtrain = [k for k in train_data['audio_embedding']]#train data
ytrain = train_data['is_turkey'].values #train data values

xval = [k for k in val_data['audio_embedding']] # validation data
yval = val_data['is_turkey'].values #validation data values

# Pad the audio features so that all are "10 seconds" long
x_train = pad_sequences(xtrain, maxlen=10) # shape (896, 10,128)
x_val = pad_sequences(xval, maxlen=10) #(299,10,128)
y_train = np.asarray(ytrain) #(896,)
y_val = np.asarray(yval) #(299,)

print("\n--- Create neural network model ---\n")

model = Sequential()

model.add(Conv1D(64, 3, activation='relu', input_shape=(10, 128)))
model.add(Conv1D(64, 3, activation='relu'))

#model.add(LSTM(20, input_shape=(10, 1), return_sequences=True))
model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(10, 1)))
#model.add(Bidirectional(CuDNNGRU(128, return_sequences = True)))

model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))

#model.add(Dense(1,activation='relu'))
model.add(Conv1D(100, 3, activation='relu'))

model.add(GlobalAveragePooling1D())

#model.add(Flatten())
#model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

print(model.summary())

# fit on a portion of the training data, and validate on the rest
from keras.callbacks import EarlyStopping, ModelCheckpoint

print("\n--- Fit the model ---\n")

# Hyper-parameters
BATCH_SIZE = 400
EPOCHS = 100

# The EarlyStopping callback monitors training accuracy:
# if it fails to improve for 20 consecutive epochs,
# training stops early
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc',mode='auto', min_delta=0.01, patience=10) ]

# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
history = model.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks= callbacks_list,
                      validation_data=[x_val, y_val],
                      verbose=1)

print("\n--- Learning curve of model training ---\n")

# summarize history for accuracy and loss
plt.figure(figsize=(6, 4))
plt.plot(history.history['acc'], "g--", label="Accuracy of training data")
plt.plot(history.history['val_acc'], "g", label="Accuracy of validation data")
plt.plot(history.history['loss'], "r--", label="Loss of training data")
plt.plot(history.history['val_loss'], "r", label="Loss of validation data")
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()
