!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/lenna.png -O lenna.png
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/baboon.png -O baboon.png
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/barbara.png -O barbara.png  

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def plot_image(image_1, image_2,title_1="Original",title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1,cmap="gray")
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2,cmap="gray")
    plt.title(title_2)
    plt.show()

image = Image.open("lenna.png")
plt.imshow(image)
plt.show()

width, height = image.size
new_width = 2 * width
new_hight = height
new_image = image.resize((new_width, new_hight))
plt.imshow(new_image)
plt.show()

new_width = width
new_hight = 2 * height
new_image = image.resize((new_width, new_hight))
plt.imshow(new_image)
plt.show()

new_width = 2 * width
new_hight = 2 * height
new_image = image.resize((new_width, new_hight))
plt.imshow(new_image)
plt.show()

new_width = width // 2
new_hight = height // 2

new_image = image.resize((new_width, new_hight))
plt.imshow(new_image)
plt.show()

theta = 45
new_image = image.rotate(theta)

plt.imshow(new_image)
plt.show()

image = np.array(image)

new_image = image + 20
plt.imshow(new_image)
plt.show()

new_image = 10 * image
plt.imshow(new_image)
plt.show()

Noise = np.random.normal(0, 20, (height, width, 3)).astype(np.uint8)
Noise.shape

new_image = image + Noise

plt.imshow(new_image)
plt.show()

new_image = image * Noise

plt.imshow(new_image)
plt.show()

im_gray = Image.open("barbara.png")

from PIL import ImageOps 

im_gray = ImageOps.grayscale(im_gray) 

im_gray = np.array(im_gray )

plt.imshow(im_gray,cmap='gray')
plt.show()

U, s, V = np.linalg.svd(im_gray , full_matrices=True)

s.shape

S = np.zeros((im_gray.shape[0], im_gray.shape[1]))
S[:image.shape[0], :image.shape[0]] = np.diag(s)

plot_image(U, V, title_1="Matrix U", title_2="Matrix V")

plt.imshow(S, cmap='gray')
plt.show()

B = S.dot(V)
plt.imshow(B,cmap='gray')
plt.show()

A = U.dot(B)

plt.imshow(A,cmap='gray')
plt.show()

for n_component in [1,10,100,200, 500]:
    S_new = S[:, :n_component]
    V_new = V[:n_component, :]
    A = U.dot(S_new.dot(V_new))
    plt.imshow(A,cmap='gray')
    plt.title("Number of Components:"+str(n_component))
    plt.show()
