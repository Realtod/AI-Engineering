import numpy as np
import matplotlib.pyplot as plt

import cv2

from skillsnetwork import cvstudio

# Initialize the CV Studio Client
cvstudioClient = cvstudio.CVStudio()

annotations = cvstudioClient.get_annotations()

model_details = cvstudioClient.downloadModel()

k_best = model_details["k_best"]

## get the model from the cv studio storage
fs = cv2.FileStorage(model_details['filename'], cv2.FILE_STORAGE_READ)
knn_yml = fs.getNode('opencv_ml_knn')

knn_format = knn_yml.getNode('format').real()
is_classifier = knn_yml.getNode('is_classifier').real()
## number of samples
default_k = knn_yml.getNode('default_k').real()
## sample arrays
samples = knn_yml.getNode('samples').mat()
## labels of sample
responses = knn_yml.getNode('responses').mat()
fs.release
knn = cv2.ml.KNearest_create()
knn.train(samples,cv2.ml.ROW_SAMPLE,responses)

my_image = cv2.imread("puppy.png")
## let's see what the image looks like
image = cv2.cvtColor(my_image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()

#make images gray
my_image = cv2.cvtColor(my_image,cv2.COLOR_BGR2GRAY)

my_image = cv2.resize(my_image, (32, 32))

pixel_image = my_image.flatten()
pixel_image = np.array([pixel_image]).astype('float32')

ret,result,neighbours,dist = knn.findNearest(pixel_image,k=k_best)
print(neighbours)
print('Your image was classified as a ' + str(annotations['labels'][int(ret)]))
