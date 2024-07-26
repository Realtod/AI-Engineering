import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics

from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())
path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv'
await download(path, "Weather_Data.csv")
filename ="Weather_Data.csv"
df = pd.read_csv("Weather_Data.csv")
df.head()

df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
df_sydney_processed.replace(['No', 'Yes'], [0,1], inplace=True)

df_sydney_processed.drop('Date',axis=1,inplace=True)
df_sydney_processed = df_sydney_processed.astype(float)
features = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
Y = df_sydney_processed['RainTomorrow']

x_train, x_test, y_train, y_test = train_test_split( features,Y, test_size=0.2, random_state=10)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)

LinearReg = LinearRegression()
x = np.asanyarray(x_train)
y = np.asanyarray(y_train)
LinearReg.fit (x, y)

# The coefficients
print ('Coefficients: ', LinearReg.coef_)

predictions = LinearReg.predict(x_test)
x = np.asanyarray(x_test)
y = np.asanyarray(y_test)
print("Residual sum of squares: %.2f"
      % np.mean((predictions - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % LinearReg.score(x, y))

from sklearn.metrics import r2_score
LinearRegression_MAE = np.mean(np.absolute(predictions - y_test))
LinearRegression_MSE = np.mean((predictions -y_test)**2)
LinearRegression_R2 = r2_score(y_test, predictions)
print("Mean absolute error: %.2f" % LinearRegression_MAE)
print("Residual sum of squares (MSE): %.2f" % LinearRegression_MSE)
print("R2-score: %.2f" % LinearRegression_R2 )

dict = {'error_type':['LinearRegression_MAE','LinearRegression_MSE','LinearRegression_R2'],'value':[LinearRegression_MAE,LinearRegression_MSE,LinearRegression_R2]}
from tabulate import tabulate
Report = pd.DataFrame(dict)
print(tabulate(Report, headers = 'keys', tablefmt = 'psql'))

#Train Model 
kNN = 4
neigh = KNeighborsClassifier(n_neighbors = kNN).fit(x_train,y_train)
neigh

predictions = neigh.predict(x_test)
predictions[0:5]

KNN_Accuracy_Score = metrics.accuracy_score(y_test, predictions)
KNN_JaccardIndex = metrics.jaccard_score(y_test, predictions)
KNN_F1_Score = metrics.f1_score(y_test, predictions)
KNN_Log_Loss = metrics.log_loss(y_test, predictions)
print("KNN Accuracy Score: ",KNN_Accuracy_Score)
print("KNN_JaccardIndex: ",KNN_JaccardIndex)
print("KNN F1 score : ", KNN_F1_Score)
print("KNN Log Loss : ", KNN_Log_Loss)

Tree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
Tree.fit(x_train, y_train)

predictions = Tree.predict(x_test)

Tree_Accuracy_Score = metrics.accuracy_score(y_test, predictions)
Tree_JaccardIndex = metrics.jaccard_score(y_test, predictions)
Tree_F1_Score = metrics.f1_score(y_test, predictions)
Tree_Log_Loss = metrics.log_loss(y_test, predictions)
print("Tree accur_acy score: ", Tree_Accuracy_Score)
print("Tree JaccardIndex : ", Tree_JaccardIndex)
print("Tree_F1_Score : ", Tree_F1_Score)
print("Tree Log Loss : ", Tree_Log_Loss)

x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size = 0.2, random_state =1)

print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)

LR = LogisticRegression(C=0.01, solver='liblinear').fit(x_train,y_train)

predictions = LR.predict(x_test)

LR_Accuracy_Score = metrics.accuracy_score(y_test,predictions)
LR_JaccardIndex = metrics.jaccard_score(y_test,predictions)
LR_F1_Score = metrics.f1_score(y_test,predictions)
LR_Log_Loss = metrics.log_loss(y_test, predictions)
print("LR accuracy score: ", LR_Accuracy_Score)
print("LR JaccardIndex : ", LR_JaccardIndex)
print("LR F1 Score : ", LR_F1_Score)
print("LR Log Loss : ", LR_Log_Loss)

SVM = svm.SVC(kernel='linear')
SVM.fit(x_train, y_train) 

predictions = SVM.predict(x_test)

SVM_Accuracy_Score = metrics.accuracy_score(y_test, predictions)
SVM_JaccardIndex = metrics.jaccard_score(y_test, predictions)
SVM_F1_Score = metrics.f1_score(y_test, predictions)
SVM_Log_Loss = metrics.log_loss(y_test, predictions)
print("SVM accuracy score : ", SVM_Accuracy_Score)
print("SVM jaccardIndex : ", SVM_JaccardIndex)
print("SVM F1_score : ", SVM_F1_Score)
print("SVM Log Loss : ", SVM_Log_Loss)

d = {'KNN':[KNN_Accuracy_Score,KNN_JaccardIndex,KNN_F1_Score,KNN_Log_Loss],
     'Tree':[Tree_Accuracy_Score, Tree_JaccardIndex, Tree_F1_Score, Tree_Log_Loss],
     'LR':[LR_Accuracy_Score, LR_JaccardIndex, LR_F1_Score,LR_Log_Loss],
     'SVM':[SVM_Accuracy_Score, SVM_JaccardIndex, SVM_F1_Score, SVM_Log_Loss]}
Report = pd.DataFrame(data=d, index = ['Accuracy','Jaccard Index','F1-Score', 'LogLoss'])
print(tabulate(Report, headers = 'keys', tablefmt = 'psql'))

