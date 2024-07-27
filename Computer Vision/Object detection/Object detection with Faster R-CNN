! wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-Coursera/images%20/images_part_5/DLguys.jpeg
! wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-Coursera/images%20/images_part_5/watts_photos2758112663727581126637_b5d4d192d4_b.jpeg
! wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-Coursera/images%20/images_part_5/istockphoto-187786732-612x612.jpeg
! wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-Coursera/images%20/images_part_5/jeff_hinton.png

! conda install pytorch=1.1.0 torchvision -c pytorch -y

import torchvision
from torchvision import transforms 
import torch
from torch import no_grad

import requests

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_predictions(pred, threshold=0.8, objects=None):
    """
    This function will assign a string name to a predicted class and eliminate predictions whose likelihood  is under a threshold 
    
    pred: a list where each element contains a tuple that corresponds to information about  the different objects; Each element includes a tuple with the class yhat, probability of belonging to that class and the coordinates of the bounding box corresponding to the object 
    image : frozen surface
    predicted_classes: a list where each element contains a tuple that corresponds to information about  the different objects; Each element includes a tuple with the class name, probability of belonging to that class and the coordinates of the bounding box corresponding to the object 
    thre
    """

    predicted_classes = [(COCO_INSTANCE_CATEGORY_NAMES[i], p, [(box[0], box[1]), (box[2], box[3])]) for i, p, box in zip(list(pred[0]['labels'].numpy()), pred[0]['scores'].detach().numpy(), list(pred[0]['boxes'].detach().numpy()))]
    predicted_classes = [stuff for stuff in predicted_classes if stuff[1] > threshold]
    
    if objects:
        predicted_classes = [stuff for stuff in predicted_classes if stuff[0] in objects]
    
    return predicted_classes

def draw_box(predicted_classes, img, rect_th=2, text_size=1, text_th=2):
    """
    This function will draw a box around the detected object and label it
    
    predicted_classes: a list where each element contains a tuple that corresponds to information about  the different objects; Each element includes a tuple with the class name, probability of belonging to that class and the coordinates of the bounding box corresponding to the object 
    image : frozen surface
    rect_th: rectangle thickness
    text_size: text size
    text_th: text thickness
    """
    img = img.permute(1, 2, 0).numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for predicted_class in predicted_classes:
        label = predicted_class[0]
        probability = predicted_class[1]
        coordinates = predicted_class[2]

        cv2.rectangle(img, coordinates[0], coordinates[1], color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img, label, coordinates[0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 255), thickness=text_th)
        cv2.putText(img, str(round(probability, 3)), (coordinates[0][0], coordinates[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 0, 0), thickness=text_th)

    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

# Load COCO labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Define a transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load a pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define a function to save RAM by deleting large variables
def save_RAM(image_=False):
    del image
    if image_:
        del img

# Download sample images
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-Coursera/images%20/images_part_5/DLguys.jpeg
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-Coursera/images%20/images_part_5/watts_photos2758112663727581126637_b5d4d192d4_b.jpeg
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-Coursera/images%20/images_part_5/istockphoto-187786732-612x612.jpeg

# Load and transform the image
img_path = 'DLguys.jpeg'
image = Image.open(img_path)
image.resize([int(0.5 * s) for s in image.size])
plt.imshow(np.array(image))
plt.show()
del img_path

img = transform(image)
pred = model([img])

# Get and draw predictions with a threshold of 0.01
pred_thresh = get_predictions(pred, threshold=0.01)
draw_box(pred_thresh, img, rect_th=1, text_size=0.5, text_th=1)
del pred_thresh

save_RAM(image_=True)

# Load and transform another image
img_path = 'istockphoto-187786732-612x612.jpeg'
image = Image.open(img_path)
image.resize([int(0.5 * s) for s in image.size])
plt.imshow(np.array(image))
plt.show()
del img_path

img = transform(image)
pred = model([img])

# Get and draw predictions with a threshold of 0.97
pred_thresh = get_predictions(pred, threshold=0.97)
draw_box(pred_thresh, img, rect_th=1, text_size=1, text_th=1)
del pred_thresh

save_RAM(image_=True)

# Example URL for loading image
url = 'https://www.plastform.ca/wp-content/themes/plastform/images/slider-image-2.jpg'

image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
del url

img = transform(image)
pred = model([img])

# Get and draw predictions with a threshold of 0.95
pred_thresh = get_predictions(pred, threshold=0.95)
draw_box(pred_thresh, img)
del pred_thresh

save_RAM(image_=True)
# img_path = 'Replace with the name of your image as seen in your directory'
# image = Image.open(img_path) # Load the image
# plt.imshow(np.array(image))
# plt.show()

# img = transform(image)
# pred = model(img.unsqueeze(0))
# pred_thresh = get_predictions(pred, threshold=0.95)
# draw_box(pred_thresh, img)
