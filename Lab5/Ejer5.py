"""
	median-filter-for-noise-removal.py

	add a description of your code here

	author: add your fullname
	date created: add this info
	universidad de monterrey
"""

# import required libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2

# read image
img_name = 'Figs/cavalo-motorizado.jpg'
img = cv2.imread(img_name)

# verify that image `img` exist
if img is None:
	print('ERROR: image ', img_name, 'could not be read')
	exit()

# define level of salt & pepper noise
s_vs_p = 0.2								
amount = 0.2								# <--- change this value

# create a copy of input image
out = img.copy()

# Generate Salt '1' noise
num_salt = np.ceil(amount * img.size * s_vs_p)
coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
out[coords] = 255

# Generate Pepper '0' noise
num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
out[coords] = 0

# apply cv2.medianBlur() for noise removal
ksize = 3					# <--- change this value
img_median = cv2.medianBlur(out, ksize)

# plot input and blurred images
plt.figure(1)
plt.imshow(img)
plt.title('Input image')
plt.xticks([])
plt.yticks([])

plt.figure(2)
plt.imshow(out)
plt.title('Noise')
plt.xticks([])
plt.yticks([])

plt.figure(3)
plt.imshow(img_median)
plt.title('Noise')
plt.xticks([])
plt.yticks([])

plt.show()