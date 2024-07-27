import numpy as np
import matplotlib.pyplot as plt

import cv2
from sklearn.externals import joblib
from skimage.feature import hog

import os
from skillsnetwork import cvstudio

# Initialize the CV Studio Client
cvstudioClient = cvstudio.CVStudio()

annotations = cvstudioClient.get_annotations()

model_details = cvstudioClient.downloadModel()

pkl_filename = model_details['filename']

svm = joblib.load(pkl_filename) 

def run_svm(image):
    ## show the original image
    orig_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(orig_image)
    plt.show()
    print('\n')
    ## convert the image into a numpy array
    image = np.array(image).astype('uint8')
    ## resize the image to a size of choice
    image = cv2.resize(image, (64, 64))
    ## convert to grayscale to reduce the information in the picture
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ## extract H.O.G features
    hog_features, hog_image = hog(grey_image,
                          visualize=True,
                          block_norm='L2-Hys',
                          pixels_per_cell=(16, 16))
    ## convert the H.O.G features into a numpy array
    image_array = np.array(hog_features)
    ## reshape the array
    image_array = image_array.reshape(1, -1)
    ## make a prediction
    svm_pred = svm.predict(image_array)
    ## print the classifier
    print('Your image was classified as a ' + str(annotations['labels'][int(svm_pred[0])]))    

## replace "your_uploaded_file" with your file name
my_image = cv2.imread("cat.jpeg")
## run the above function on the image to get a classification
run_svm(my_image)
