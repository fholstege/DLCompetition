from PIL import Image
from torchvision import transforms
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, misc

# define path to image parent directory
image_path = Path(os.getcwd())

# load image as grayscale with alpha values
image = Image.open(image_path/"triang.png").convert('LA')

# check size
image.size

# create transformation to Pad image
img_transform = transforms.Compose([
    transforms.Pad(1, fill = 255)
])

padded_image = img_transform(image)

# check size
print(padded_image.size)

# convert to numpy array
img_array = np.array(padded_image)
print(img_array.shape)

# code prewitt filter
kernel_x = np.array([[1,0,-1],
                     [1,0,-1],
                     [1,0,-1]])
kernel_y = kernel_x.T

# function that applies a 3x3 filter
def apply_filter(img, filter):
    # get shape dimensions
    x,y,z = img.shape

    # init convolution
    C = np.zeros((x-2,y-2,z))

    # compute Gx going over z, y, x dimensions
    for k in range(z):
        for j in range(y-2):
            for i in range(x-2):
                # print(i,j,k)
                C[i,j,k] = np.sum(np.multiply(filter, img[i:i+3, j:j+3, k]))
                
    return C

# apply filters
Gx = apply_filter(img_array, kernel_x)
print(Gx.shape)
Image.fromarray(Gx.astype(np.uint8)) # integer otherwise cannot print image

Gy = apply_filter(img_array, kernel_y)
print(Gy.shape)
Image.fromarray(Gy.astype(np.uint8))

# apply formula from wikipedia to combine kernels
G = np.sqrt(np.power(Gx, 2) + np.power(Gy, 2))

# scale pixels back to a maximum of 255
G = G/np.max(G) * 255

# convert to integers
G = G.astype(np.uint8)

# print image
plt.figure(figsize = (6,6))
plt.imshow(G[:,:,1], cmap='gray')
plt.tight_layout()
plt.savefig(image_path/"prewitt_img.png", dpi = 320)