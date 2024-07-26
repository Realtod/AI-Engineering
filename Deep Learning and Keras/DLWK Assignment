import pandas as pd
import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense

# load the concrete_data dataset
concrete_data = pd.read_csv('https://cocl.us/concrete_data')

# verify that the data was loaded correctly
concrete_data.head()

# split the concrete_data set into predictors (inputs) and target (output)
predictors = concrete_data.drop(columns=['Strength'])
target = concrete_data['Strength']

n_cols = predictors.shape[1]

# Create the neural network in a function so we can use it multiple times in the
# subsequent sections
def regression_model():
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = regression_model()

from sklearn.model_selection import train_test_split

predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, target, 
                                                                                test_size=0.3, random_state=71)
# let's have a look at the shape of the predictors set
predictors_train.shape

model.fit(predictors_train, target_train, epochs=50)

predictions_test = model.predict(predictors_test)
from sklearn.metrics import mean_squared_error
print('Mean squared error on test data is %.3f' % (mean_squared_error(target_test, predictions_test)))


# Create a function that evaluates the model so we can use it to evaluate the models created in part A, B, C and D
# the 'create_model_func' parameter is the function that is used to build the model. For part A, this is the
# regression_model function defined above
def evaluate_model(create_model_func, predictors, targets, epochs=50):
    mean_squared_errors = []
    for i in range(50):
        # create the model. I wasn't 100% clear whether this should be inside the loop, but I _think_ that was the
        # intent of the question. Otherwise, the average and stddev of the mean squared error is not that meaningful
        model = create_model_func()
        # 1. split the data in a train and test set
        predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, target, 
                                                                                    test_size=0.3, random_state=71)
        # 2. train 50 epochs (suppress logging this time)
        model.fit(predictors_train, target_train, epochs=epochs, verbose=0)
        # 3. measure the mse and add this to the list
        predictions_test = model.predict(predictors_test)
        mse = mean_squared_error(target_test, predictions_test)
        mean_squared_errors.append(mse)
        print('.', end='') # output a dot so we can see that the function is still running
    print(' Done!')
    # return the mean and stddev of the mse list
    return np.mean(mean_squared_errors), np.std(mean_squared_errors)
# Evaluate the model and print the mean and std dev of the mean squared errors. Note that we pass in 
# the regression_model _function_ here. This is used in the evaluate_model function to create a fresh
# neural network in each loop
mean_mse, std_mse = evaluate_model(regression_model, predictors, target)
# Report the mean and stddev of the mean squared errors
print("Mean squared errors for 50 regression models: mean = %.3f, std dev = %.3f" %(mean_mse, std_mse))

# Normalize the data
predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()

mean_mse, std_mse = evaluate_model(regression_model, predictors_norm, target)
# Report the mean and stddev of the mean squared errors
print("Mean squared errors for 50 regression models on normalized data: mean = %.3f, std dev = %.3f" % 
      (mean_mse, std_mse))

mean_mse, std_mse = evaluate_model(regression_model, predictors_norm, target, epochs=100)
# Report the mean and stddev of the mean squared errors
print("Mean squared errors for 50 regression models on normalized data, trained 100 epochs: mean = %.3f, std dev = %.3f" % 
      (mean_mse, std_mse))
mean_mse, std_mse = evaluate_model(regression_model, predictors_norm, target, epochs=100)
# Report the mean and stddev of the mean squared errors
print("Mean squared errors for 50 regression models on normalized data, trained 100 epochs: mean = %.3f, std dev = %.3f" % 
      (mean_mse, std_mse))
