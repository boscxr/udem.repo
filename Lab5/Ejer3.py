"""
	blur-image.py

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
img_name = 'cavalo-motorizado.jpg'
img = cv2.imread(img_name)

# verify that image `img` exist
if img is None:
	print('ERROR: image ', img_name, 'could not be read')
	exit()

# blur image using `cv2.blur()`
kernel_size = (11,11)
blurred_image = cv2.blur(img, kernel_size)

# plot input and blurred images
plt.figure(1)
plt.imshow(img)
plt.title('Input image')
plt.xticks([])
plt.yticks([])

plt.figure(2)
plt.imshow(blurred_image)
plt.title('Output image using cv2.blur(%i,%i)' % (kernel_size[0], kernel_size[1]))
plt.xticks([])
plt.yticks([])

plt.show()