from js import fetch
import io

URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/real_estate_data.csv"
resp = await fetch(URL)
regression_tree_data = io.BytesIO((await resp.arrayBuffer()).to_py())

import piplite
await piplite.install(['pandas'])
await piplite.install(['numpy'])
await piplite.install(['scikit-learn'])

# Pandas will allow us to create a dataframe of the data so it can be used and manipulated
import pandas as pd
# Regression Tree Algorithm
from sklearn.tree import DecisionTreeRegressor
# Split our data into a training and testing data
from sklearn.model_selection import train_test_split

data = pd.read_csv(regression_tree_data)
data.head()

data.isna().sum()
data.dropna(inplace=True)
data.isna().sum()

X = data.drop(columns=["MEDV"])
Y = data["MEDV"]
X.head()
Y.head()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=1)

regression_tree = DecisionTreeRegressor(criterion = "mse")
regression_tree.fit(X_train, Y_train)

regression_tree.score(X_test, Y_test)
prediction = regression_tree.predict(X_test)
print("$",(prediction - Y_test).abs().mean()*1000)

regression_tree = DecisionTreeRegressor(criterion = "mae")
regression_tree.fit(X_train, Y_train)
print(regression_tree.score(X_test, Y_test))
prediction = regression_tree.predict(X_test)
print("$",(prediction - Y_test).abs().mean()*1000)
