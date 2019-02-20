"""
    measure_code_performance.py

    add a description of your code here

    author: add your fullname
    date created: add this info
    universidad de monterrey
"""

# import required libraries
import numpy as numpy
import matplotlib.pyplot as plt
import cv2

# print image statistics
def print_image_statistics(img, head_string):

    """
        include docstring here
    """
    # add your code below this line

    return None

# visualise image
def visualise_image(img, fig_number, fig_title, iscolour):

    """
        include docstring here
    """
    # add your code below this line

    return None

# read image
image_name = 'figs/vehicular_traffic.jpg'
img_colour = cv2.imread(image_name, cv2.IMREAD_COLOR)

# verify that image exists
if img_colour is None:
    print('ERROR: image ', image_name, 'could not be read')
    exit()

# convert from BGR to RGB so that the image can be visualised using matplotlib
img_colour = cv2.cvtColor(img_colour, cv2.COLOR_BGR2RGB)

# convert the input colour image into a grayscale image
e1 = cv2.getTickCount()
img_greyscale = cv2.cvtColor(img_colour, cv2.COLOR_BGR2GRAY)
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print('\nConversion from colour to greyscale took: ', time, 'seconds')

# print colour image stats and visualise it
e1 = cv2.getTickCount()
print_image_statistics(img_colour, 'COLOUR IMAGE STATS:')
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print('\nCODE PERFORMANCE:')
print('printing colour image stats took: ', time, 'seconds')

e1 = cv2.getTickCount()
visualise_image(img_colour, 1, 'INPUT IMAGE: COLOUR', 1)
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print('visualising colour image took: ', time, 'seconds')

# print greyscale image stats and visualise it
e1 = cv2.getTickCount()
print_image_statistics(img_greyscale, 'GREYSCALE IMAGE STATS:')
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print('\nCODE PERFORMANCE:')
print('Printing greyscale image stats took: ', time, 'seconds')

e1 = cv2.getTickCount()
visualise_image(img_greyscale, 2, 'OUTPUT IMAGE: GREYSCALE', 0)
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print('Visualising grey colour image took: ', time, 'seconds')

# visualise figures
plt.show()