import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
     

concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()


concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

# Split the dataset into training and test 
X_train, X_test, y_train, y_test = train_test_split( predictors, target, test_size=0.3, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

n_cols = predictors.shape[1] # number of predictors
print('Number of predictors:', n_cols)


# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
     

# build the model
model = regression_model()

# fit the model
model.fit(X_train, y_train, epochs=50, verbose=1)

# predict using X_test
y_pred = model.predict(X_test)

loss = mean_squared_error(y_test, y_pred)
mean = np.mean(loss)
standard_deviation = np.std(loss)

print("Mean Squared error:", mean)
print("Standar Deviation:", standard_deviation)



# Repeat 50 times
total_mean_squared_errors = 50
epochs = 50
mean_squared_errors = []
for i in range(0, total_mean_squared_errors):
    # Split the dataset into training and test 
    X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=i)

    # fit the model
    model.fit(X_train, y_train, epochs=epochs, verbose=0)

    # evaluate the model
    MSE = model.evaluate(X_test, y_test, verbose=0)
    print("MSE "+str(i+1)+": "+str(MSE))

    # predict using X_test
    y_pred = model.predict(X_test)
    mean_square_error = mean_squared_error(y_test, y_pred)
    mean_squared_errors.append(mean_square_error)

mean_squared_errors = np.array(mean_squared_errors)
mean = np.mean(mean_squared_errors)
standard_deviation = np.std(mean_squared_errors)

print('\n')
print("Below is the mean and standard deviation of " +str(total_mean_squared_errors) + " mean squared errors without normalized data. Total number of epochs for each training is: " +str(epochs) + "\n")
print("Mean: "+str(mean))
print("Standard Deviation: "+str(standard_deviation))
