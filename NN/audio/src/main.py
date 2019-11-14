# https://towardsdatascience.com/getting-started-on-deep-learning-for-audio-data-667d9aa76a33
# https://www.tensorflow.org/lite/convert/python_api

import numpy as np
import pandas as pd 
import os
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pdb
from tqdm import tqdm
from sklearn.model_selection import KFold, cross_val_score, train_test_split

plt.style.use('fivethirtyeight')
print(os.listdir("../input"))

# train = pd.read_json('../input/train.json')
train = pd.read_json('../../data/1py/data.json')

train_train, train_val = train_test_split(train, random_state = 42) # Split into training set and validation set

xtrain = [k for k in train_train['fft']]
ytrain = train_train['crow'].values
xval = [k for k in train_val['fft']]
yval = train_val['crow'].values

# Pad the audio features so that all are "10 seconds" long

x_train = tf.keras.preprocessing.sequence.pad_sequences(xtrain)
x_val = tf.keras.preprocessing.sequence.pad_sequences(xval)
y_train = np.asarray(ytrain)
y_val = np.asarray(yval)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1,input_shape=(128,)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer = tf.keras.optimizers.Nadam(lr=0.001), metrics=['accuracy'])
print(model.summary())
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

# fit on a portion of the training data, and validate on the rest
# from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=2, verbose=1, min_lr=1e-8)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=20,  restore_best_weights=True)
history = model.fit(x_train, y_train,batch_size=512, epochs=1000, validation_data=[x_val, y_val],verbose = 2,callbacks=[reduce_lr,early_stop])

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
