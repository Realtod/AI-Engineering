import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from imutils import paths
import seaborn as sns
import random
import time
from datetime import datetime

import cv2
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

import os
from skillsnetwork import cvstudio

def load_images(image_paths):
# loop over the input images
    for (i, image_path) in enumerate(image_paths):
        #read image
        image = cv2.imread(image_path)
        image = np.array(image).astype('uint8')
        image = cv2.resize(image, (64, 64))
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog_features, hog_images = hog(grey_image,
                              visualize=True,
                              block_norm='L2-Hys',
                              pixels_per_cell=(16, 16))
        #label image using the annotations
        label = class_object.index(annotations["annotations"][image_path[7:]][0]['label'])
        train_images.append(hog_features)
        train_labels.append(label)

# Initialize the CV Studio Client
cvstudioClient = cvstudio.CVStudio()

# Download All Images
cvstudioClient.downloadAll()

annotations = cvstudioClient.get_annotations()

first_five = {k: annotations["annotations"][k] for k in list(annotations["annotations"])[:5]}
first_five

sample_image = 'images/' + random.choice(list(annotations["annotations"].keys()))

sample_image = cv2.imread(sample_image)

sample_image = cv2.resize(sample_image, (64, 64))
sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)

plt.imshow(sample_image, cmap=plt.cm.gray)

## when we run H.O.G., it returns an array of features and the image/output it produced
## the feature is what we use to train the SVM model
sample_image_features, sample_hog_image = hog(sample_image,
                              visualize=True,
                              block_norm='L2-Hys',
                              pixels_per_cell=(16, 16))

## lets look at what the H.O.G. feature looks like
plt.imshow(sample_hog_image, cmap=plt.cm.gray)

image_paths = list(paths.list_images('images'))
train_images = []
train_labels = []
class_object = annotations['labels']

load_images(image_paths)

train_array = np.array(train_images)
train_array = np.vstack(train_array)

labels_array = np.array(train_labels)

labels_array = labels_array.astype(int)
labels_array = labels_array.reshape((labels_array.size,1))

train_df = np.concatenate([train_array, labels_array], axis = 1)

percentage = 75
partition = int(len(train_df)*percentage/100)

x_train, x_test = train_df[:partition,:-1],  train_df[partition:,:-1]
y_train, y_test = train_df[:partition,-1:].ravel(), train_df[partition:,-1:].ravel()

param_grid = {'kernel': ('linear', 'rbf'),'C': [1, 10, 100]}

base_estimator = SVC(gamma='scale')

start_datetime = datetime.now()
start = time.time()

svm = GridSearchCV(base_estimator, param_grid, cv=5)
#Fit the data into the classifier
svm.fit(x_train,y_train)
#Get values of the grid search
best_parameters = svm.best_params_
print(best_parameters)
#Predict on the validation set
y_pred = svm.predict(x_test)
# Print accuracy score for the model on validation  set. 
print("Accuracy: "+str(accuracy_score(y_test, y_pred)))

end = time.time()
end_datetime = datetime.now()
print(end - start)

label_names = [0, 1]
cmx = confusion_matrix(y_test, y_pred, labels=label_names)

df_cm = pd.DataFrame(cmx)
# plt.figure(figsize=(10,7))
sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
title = "Confusion Matrix for SVM results"
plt.title(title)
plt.show()

parameters = {
    'best_params': best_parameters
}
result = cvstudioClient.report(started=start_datetime, completed=end_datetime, parameters=parameters, accuracy=accuracy_score(y_test, y_pred))

if result.ok:
    print('Congratulations your results have been reported back to CV Studio!')

filename = 'finalized_model.sav'
joblib.dump(svm, filename)y_score(y_test, y_pred))

if result.ok:
    print('Congratulations your results have been reported back to CV Studio!')

# Save the SVM model to a file
joblib.dump(svm.best_estimator_, 'svm.joblib')

# Now let's save the model back to CV Studio
result = cvstudioClient.uploadModel('svm.joblib', {'svm_best': svm.best_estimator_})
