import pandas as pd
import numpy  as np

concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()

concrete_data.describe()

concrete_data.isnull().sum()

concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

predictors.head()

predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()

n_cols = predictors_norm.shape[1] # number of predictors
print("n predictions = {}".format(n_cols))

import keras
from keras.models import Sequential
from keras.layers import Dense

# define regression model
def regression_model(layers_cfg,input_shape):
    # create model
    model = Sequential()
    
    nlayers     = len(layers_cfg)
    final_layer = "layer_" + str(nlayers)
    
    for k,layer in layers_cfg.items():
        n_units    = layer.get("n_units")
        activation = layer.get("activation")
        if k == "layer_1":
            model.add(Dense(n_units, activation=activation, input_shape=input_shape))
        elif k == final_layer:
            model.add(Dense(n_units))
        else:
            model.add(Dense(n_units, activation=activation))
    
    # model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# build the model

network_config = {"layer_1": {"n_units": 50, "activation": "relu"},
                  "layer_2": {"n_units": 50, "activation": "relu"},
                  "layer_3": {"n_units":  1},
                 }
input_shape = (n_cols,)

print("input_shape    = {}".format(input_shape))
print("network_config = {}".format(network_config))

model = regression_model(network_config,input_shape)
model.summary()
# model = regression_model()

# fit the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)

