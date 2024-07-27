### install opencv version 3.4.2 for this exercise, 
### if you have a different version of OpenCV please switch to the 3.4.2 version
# !{sys.executable} -m pip install opencv-python==3.4.2.16
import urllib.request
import cv2
print(cv2.__version__)
from matplotlib import pyplot as plt
%matplotlib inline

def plt_show(image, title="", gray=False, size=(12,10)):
    from pylab import rcParams
    temp = image 
    
    # Convert to grayscale images
    if gray == False:
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    
    # Change image size
    rcParams['figure.figsize'] = [10,10]
    # Remove axes ticks
    plt.axis("off")
    plt.title(title)
    plt.imshow(temp, cmap='gray')
    plt.show()

def detect_obj(image):
    # Clean your image
    plt_show(image)
    ## Detect the car in the image
    object_list = detector.detectMultiScale(image)
    print(object_list)
    # For each car, draw a rectangle around it
    for obj in object_list: 
        (x, y, w, h) = obj
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2) # Line thickness
    ## Lets view the image
    plt_show(image)

## Read the url
haarcascade_url = 'https://raw.githubusercontent.com/andrewssobral/vehicle_detection_haarcascades/master/cars.xml'
haar_name = "cars.xml"
urllib.request.urlretrieve(haarcascade_url, haar_name)

detector = cv2.CascadeClassifier(haar_name)

## We will read in a sample image
image_url = "https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/CV0101/Dataset/car-road-behind.jpg"
image_name = "car-road-behind.jpg"
urllib.request.urlretrieve(image_url, image_name)
image = cv2.imread(image_name)

plt_show(image)

detect_obj(image)

## Replace "your_uploaded_file" with your file name
my_image = cv2.imread("your_uploaded_file")

detect_obj(my_image)
