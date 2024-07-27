import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

concrete_data = pd.read_csv('concrete_data.csv')
dataset = concrete_data.values
# split into input (X) and output (Y) variables
X = dataset[:,0:7]
Y = dataset[:,8]

# Create the neural network in a function so we can use it multiple times in the
# subsequent sections
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(7, input_dim=7, init='normal', activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

model = baseline_model()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=71)
# let's have a look at the shape of the predictors set
X_train.shape

model.fit(X_train, y_train, epochs=50)

y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print('Mean squared error on test data is %.3f' % (mean_squared_error(y_test, y_pred)))

mse = []
for i in range(50):
    model = baseline_model()
    model.fit(X_train, y_train, epochs=50)
    y_pred = model.predict(X_test)
    mse.append(mean_squared_error(y_test, y_pred))

dataset_norm = []
for i in range(9):
    _mean = np.mean(dataset[:,i])
    _std = np.std(dataset[:,i])
    _norm = (dataset[:,i] - _mean) / _std
    dataset_norm.append(_norm)
dataset_normalized = np.asarray(dataset_norm).T
# split the concrete_data set into predictors (inputs) and target (output)
X = dataset_normalized[:,0:7]
y = dataset_normalized[:,8]

# Create the neural network in a function so we can use it multiple times in the
# subsequent sections
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(7, input_dim=7, init='normal', activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

model = baseline_model()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=71)
# let's have a look at the shape of the predictors set
X_train.shape

model.fit(X_train, y_train, epochs=50)

y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print('Mean squared error on test data is %.3f' % (mean_squared_error(y_test, y_pred)))

mse = []
for i in range(50):
    model = baseline_model()
    model.fit(X_train, y_train, epochs=50)
    y_pred = model.predict(X_test)
    mse.append(mean_squared_error(y_test, y_pred))

dataset_norm = []
for i in range(9):
    _mean = np.mean(dataset[:,i])
    _std = np.std(dataset[:,i])
    _norm = (dataset[:,i] - _mean) / _std
    dataset_norm.append(_norm)

dataset_normalized = np.asarray(dataset_norm).T
# split the concrete_data set into predictors (inputs) and target (output)
X = dataset_normalized[:,0:7]
y = dataset_normalized[:,8]

n_cols = X.shape[1]

# Create the neural network in a function so we can use it multiple times in the
# subsequent sections
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(7, input_dim=7, init='normal', activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=71)
# let's have a look at the shape of the predictors set
X_train.shape

model.fit(X_train, y_train, epochs=50)

y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print('Mean squared error on test data is %.3f' % (mean_squared_error(y_test, y_pred)))

mse = []
for i in range(50):
    model = baseline_model()
    model.fit(X_train, y_train, epochs=100)
    y_pred = model.predict(X_test)
    mse.append(mean_squared_error(y_test, y_pred))

dataset_norm = []
for i in range(9):
    _mean = np.mean(dataset[:,i])
    _std = np.std(dataset[:,i])
    _norm = (dataset[:,i] - _mean) / _std
    dataset_norm.append(_norm)
dataset_normalized = np.asarray(dataset_norm).T
# split the concrete_data set into predictors (inputs) and target (output)
X = dataset_normalized[:,0:7]
y = dataset_normalized[:,8]

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(7, input_dim=7, init='normal', activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=71)
# let's have a look at the shape of the predictors set
X_train.shape

model.fit(X_train, y_train, epochs=50)

y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print('Mean squared error on test data is %.3f' % (mean_squared_error(y_test, y_pred)))

mse = []
for i in range(50):
    model = baseline_model()
    model.fit(X_train, y_train, epochs=50)
    y_pred = model.predict(X_test)
    mse.append(mean_squared_error(y_test, y_pred))

  
