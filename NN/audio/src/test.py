import numpy as np
import pandas as pd
import os
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import KFold, cross_val_score, train_test_split

#plt.style.use('fivethirtyeight')
test = pd.read_json('../input/train.json')

import keras
from keras import models
from keras import optimizers
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import BatchNormalization, Bidirectional, LSTM, SimpleRNN
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv1D, MaxPooling1D,GlobalMaxPooling1D
from keras.utils import np_utils

print("\n--- Create neural network model ---\n")

model = Sequential()

model.add(Conv1D(64, 3, activation='relu', input_shape=(10, 128)))
model.add(Conv1D(64, 3, activation='relu'))

#model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(10, 1)))
model.add(Bidirectional(SimpleRNN(50, return_sequences = True, activation="tanh")))

model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(100, 3, activation='relu'))

model.add(GlobalAveragePooling1D())

model.add(Dense(1, activation='sigmoid'))

#load models
from keras.models import load_model

print("\n--- Load created best model & Compile it---\n")

model = load_model('best_model.h5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

print(model.summary())

print("\n--- Check against test data ---\n")
# the format of test data should also same as training and validation data but different contents
# Otherwise it won't fit into the model
dummy_data, test_data = train_test_split(test, test_size=0.33, random_state = 42) # Split into training set and validation set

xtest = [k for k in test_data['audio_embedding']] # test data
ytest = test_data['is_turkey'].values #test data values

# Pad the audio features so that all are "10 seconds" long

x_test = pad_sequences(xtest, maxlen=10) #(395,10,128)
y_test = np.asarray(ytest) #(395,)

score = model.evaluate(x_test, y_test, verbose=1)

print("\nAccuracy on test data: %0.2f%%" % (score[1]*100))
print("\nLoss on test data: %0.2f" % score[0])
