!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/lenna.png -O lenna.png
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/baboon.png -O baboon.png
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/barbara.png -O barbara.png  

def get_concat_h(im1, im2):
    #https://note.nkmk.me/en/python-pillow-concat-images/
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

my_image = "lenna.png"

import os
cwd = os.getcwd()
cwd 

image_path = os.path.join(cwd, my_image)
image_path

import cv2

image = cv2.imread(my_image)

type(image)

image.shape

image.max()

image.min()

#cv2.imshow('image', imgage)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.imshow(image)
plt.show()

new_image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,10))
plt.imshow(new_image)
plt.show()

image = cv2.imread(image_path)
image.shape

cv2.imwrite("lenna.jpg", image)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image_gray.shape

plt.figure(figsize=(10, 10))
plt.imshow(image_gray, cmap='gray')
plt.show()

cv2.imwrite('lena_gray_cv.jpg', image_gray)

im_gray = cv2.imread('barbara.png', cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(10,10))
plt.imshow(im_gray,cmap='gray')
plt.show()

baboon=cv2.imread('baboon.png')
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.show()

blue, green, red = baboon[:, :, 0], baboon[:, :, 1], baboon[:, :, 2]

im_bgr = cv2.vconcat([blue, green, red])

plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.title("RGB image")
plt.subplot(122)
plt.imshow(im_bgr, cmap='gray')
plt.title("BGR image")
plt.show()

plt.imshow(im_bgr, cmap='gray')
plt.title("Different color channels  blue (top), green (middle), red (bottom)  ")
plt.show()

rows = 256

plt.figure(figsize=(10,10))
plt.imshow(new_image[0:rows,:,:])
plt.show()

columns = 256

plt.figure(figsize=(10,10))
plt.imshow(new_image[:,0:columns,:])
plt.show()

A = new_image.copy()
plt.imshow(A)
plt.show()

B = A
A[:,:,:] = 0
plt.imshow(B)
plt.show()

baboon_red = baboon.copy()
baboon_red[:, :, 0] = 0
baboon_red[:, :, 1] = 0
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(baboon_red, cv2.COLOR_BGR2RGB))
plt.show()

baboon_blue = baboon.copy()
baboon_blue[:, :, 1] = 0
baboon_blue[:, :, 2] = 0
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(baboon_blue, cv2.COLOR_BGR2RGB))
plt.show()

baboon_green = baboon.copy()
baboon_green[:, :, 0] = 0
baboon_green[:, :, 2] = 0
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(baboon_green, cv2.COLOR_BGR2RGB))
plt.show()

image=cv2.imread('baboon.png')

plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

image=cv2.imread('baboon.png') # replace and add you image here name 
baboon_blue=image.copy()
baboon_blue[:,:,1] = 0
baboon_blue[:,:,2] = 0
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(baboon_blue, cv2.COLOR_BGR2RGB))
plt.show()

baboon_blue = cv2.imread('baboon.png')
baboon_blue = cv2.cvtColor(baboon_blue, cv2.COLOR_BGR2RGB)
baboon_blue[:,:,2] = 0
plt.figure(figsize=(10,10))
plt.imshow(baboon_blue)
plt.show()
