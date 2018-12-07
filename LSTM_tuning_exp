import os
import numpy as np
import pandas as pd

import keras
from keras.optimizers import SGD, Adam
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
#from keras.layers import Embedding
#from keras.layers import TimeDistributed

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Loop over hidden_units and learning rate

results = []

lst_hidden_units = [35, 50, 75, 100, 150]
lr_lst = [0.00001, 0.0001, 0.001, 0.01, 0.1]

for i, hidden_unit in enumerate(lst_hidden_units):
	for j, lr_ in enumerate(lr_lst):

		# Data
		convert = np.load('num_targets.npy')
		MFCC_matrix = np.load('MFCC_matrix.npy')


		np.random.seed(1337)  # for reproducibility
		initializers.glorot_uniform(seed=1337)

		##### Settings ####
		## Set name to save files
		model_name = 'model_{0}_{1}'.format(i, j)
		batch_size = 5
		nb_classes = 35
		epochs = 15

		hidden_units = hidden_unit

		# Load and split the data
		yy = pd.Series(convert)
		xx = MFCC_matrix

		# normalize the dataset
		scaler = MinMaxScaler(feature_range=(0, 1))
		xx = np.array([scaler.fit_transform(mx) for mx in xx])

		# Split the data 
		X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size = 0.2, random_state=0)

		# Set the files to be divisible by 5
		X_train = X_train[:75855,:]
		y_train = y_train[:75855]

		# Build categorical tensors
		Y_train = np_utils.to_categorical(y_train, nb_classes)
		Y_test = np_utils.to_categorical(y_test, nb_classes)

		model = Sequential()
		# test with embed
		#model.add(Embedding(nb_classes, hidden_units, input_length=nb_classes))
		model.add(LSTM(output_dim=hidden_units, activation='tanh', return_sequences=True,
		               recurrent_activation='hard_sigmoid', use_bias=True,
		               batch_input_shape=(batch_size, 99, X_train.shape[2])))

		# model.add(Dropout(0.5)) # Not really improved on the model

		model.add(LSTM(output_dim=hidden_units, activation='tanh', 
		               return_sequences=False,
		               recurrent_activation='hard_sigmoid', use_bias=True))

		model.add(Dense(nb_classes))
		model.add(Activation('softmax'))

		# try using different optimizers and different optimizer configs
		# Adam optimizer has best results - SGD was tried
		# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

		adam = Adam(lr=lr_, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) # smaller lr than model 2
		model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

		history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, 
		                    validation_data=(X_test, Y_test), verbose=1) 

		score, acc = model.evaluate(X_test, Y_test,
		                            batch_size=batch_size, verbose=1)

		np.save('.//{}_history.npy'.format(model_name), np.array(history.history))
		np.save('.//{}_score_acc.npy'.format(model_name), np.array([score, acc]))

		# Install - sudo pip install h5py
		model.save('.//{}_LSTM.h5'.format(model_name))
		model.save_weights('.//{}_weights.h5'.format(model_name))

		print('Test score:', score)
		print('Test accuracy:', acc)

		results.append([score, acc])

np.save('.//results.npy', np.array(results))
