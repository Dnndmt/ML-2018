
# coding: utf-8

# In[1]:

import os
import numpy as np
import pandas as pd

from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM

from sklearn.model_selection import train_test_split


# In[2]:

# Data
convert = np.load('num_targets.npy')
MFCC_matrix = np.load('MFCC_matrix.npy')


# In[3]:

np.random.seed(1337)  # for reproducibility

batch_size = 5
hidden_units = 100
nb_classes = 35

print('Loading data...')
# Load and split the data
yy = pd.Series(convert)
xx = MFCC_matrix
X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size = 0.2, random_state=0)

# Set the files to be divisible by 5
X_train = X_train[:75855,:]
y_train = y_train[:75855]


print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
print('Build model...')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print(batch_size, 99, X_train.shape[2])
print(X_train.shape[1:])
print(X_train.shape[2])

model = Sequential()
model.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
               forget_bias_init='one', activation='tanh', inner_activation='sigmoid', return_sequences=True,
               stateful=True, batch_input_shape=(batch_size, 99, X_train.shape[2])))

# model.add(Dropout(0.5)) # Not really improved on the model

model.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
               forget_bias_init='one', activation='tanh', inner_activation='sigmoid', 
               return_sequences=False, stateful=True))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# try using different optimizers and different optimizer configs
# Adam optimizer was tried already
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print("Train...")
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=3, validation_data=(X_test, Y_test)) 

score, acc = model.evaluate(X_test, Y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

