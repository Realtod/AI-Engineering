import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.utils  import to_categorical

import matplotlib.pyplot as plt

# import the data
from keras.datasets import mnist

# read the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[0])

# flatten images into one-dimensional vector

num_pixels = X_train.shape[1] * X_train.shape[2] # find size of one-dimensional vector

X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32') # flatten training images
X_test  = X_test.reshape(X_test.shape[0], num_pixels).astype('float32') # flatten test images
print("X_train_flatten.shape = {}".format(X_train.shape))
print("X_test_flatten.shape  = {}".format(X_test.shape))

import numpy as np
print("min pixels train = {}".format(np.amin(X_train)))
print("max pixels train = {}".format(np.amax(X_train)))
print("min pixels test  = {}".format(np.amin(X_test)))
print("max pixels test  = {}".format(np.amax(X_test)))

print(type(X_test))
print(type(X_train))

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test  = X_test  / 255
print("y_test.shape  = {}".format(y_test.shape))
print("y_train.shape = {}".format(y_train.shape))

print(y_test[0:10])
print(y_train[0:10])

# one hot encode outputs
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

print(y_test.shape)
print(y_train.shape)

num_classes = y_test.shape[1]
print(num_classes)

# define classification model
def classification_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# build the model
model = classification_model()
model.summary()

# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)

print('Accuracy: {}% \n Error: {}'.format(scores[1], 1 - scores[1]))        

model.save('classification_model.h5')

from keras.models import load_model

pretrained_model = load_model('classification_model.h5')
