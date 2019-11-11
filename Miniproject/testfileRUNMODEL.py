import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import os
import numpy as np
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

X = np.asarray(pickle.load(open("XX.pickle", "rb")))
Y = np.asarray(pickle.load(open("YY.pickle", "rb")))

X = X/255.0

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(8))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

model.fit(X, Y, batch_size = 20, epochs = 10, validation_split = 0.3)
