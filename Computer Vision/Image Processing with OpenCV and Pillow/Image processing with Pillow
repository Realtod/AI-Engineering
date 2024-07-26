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

image_path = os.path.join(cwd, my_image)

from PIL import Image

image = Image.open(my_image)
type(image)

image.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.imshow(image)
plt.show()

image = Image.open(image_path)
print(image.size)
print(image.mode)
im = image.load() 

x = 0
y = 1
im[y,x]

image.save("lenna.jpg")

from PIL import ImageOps 

image_gray = ImageOps.grayscale(image) 
image_gray.mode

image_gray.quantize(256 // 2)
image_gray.show()

#get_concat_h(image_gray,  image_gray.quantize(256//2)).show(title="Lena") 
for n in range(3,8):
    plt.figure(figsize=(10,10))

    plt.imshow(get_concat_h(image_gray,  image_gray.quantize(256//2**n))) 
    plt.title("256 Quantization Levels  left vs {}  Quantization Levels right".format(256//2**n))
    plt.show()

baboon = Image.open('baboon.png')
red, green, blue = baboon.split()
get_concat_h(baboon, red)
get_concat_h(baboon, blue)
get_concat_h(baboon, green)

import numpy as np

array= np.asarray(image)
print(type(array))

array = np.array(image)

# summarize shape
print(array.shape)
print(array)
array[0, 0]

plt.figure(figsize=(10,10))
plt.imshow(array)
plt.show()

rows = 256

plt.figure(figsize=(10,10))
plt.imshow(array[0:rows,:,:])
plt.show()

columns = 256

plt.figure(figsize=(10,10))
plt.imshow(array[:,0:columns,:])
plt.show()

A = array.copy()
plt.imshow(A)
plt.show()

B = A
A[:,:,:] = 0
plt.imshow(B)
plt.show()
	
baboon_array = np.array(baboon)
plt.figure(figsize=(10,10))
plt.imshow(baboon_array)
plt.show()

baboon_array = np.array(baboon)
plt.figure(figsize=(10,10))
plt.imshow(baboon_array[:,:,0], cmap='gray')
plt.show()

baboon_red=baboon_array.copy()
baboon_red[:,:,1] = 0
baboon_red[:,:,2] = 0
plt.figure(figsize=(10,10))
plt.imshow(baboon_red)
plt.show()

baboon_blue=baboon_array.copy()
baboon_blue[:,:,0] = 0
baboon_blue[:,:,1] = 0
plt.figure(figsize=(10,10))
plt.imshow(baboon_blue)
plt.show()

blue_lenna = Image.open('lenna.png')
blue_array = np.array(blue_lenna)
blue_array[:, :, 2] = 0
plt.figure(figsize=(10, 10))
plt.imshow(blue_array)
plt.show()
