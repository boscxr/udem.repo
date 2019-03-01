# importa librerías estandar
import numpy as np
import cv2

# read in input image
img_in = cv2.imread('figs/vehicular-traffic.jpg', cv2.IMREAD_COLOR) # alternatively, you can use cv2.IMREAD_GRAYSCALE

# create a new window for image visualisation purposes
cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)  # alternatively, you can use cv2.WINDOW_NORMAL

# visualise input image
cv2.imshow("input image", img_in)

# convert input image from colour to greyscale
img_out = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

# visualise greyscale image
cv2.imshow("greyscale image", img_out)

# wait for the user to press a key
key = cv2.waitKey(0)

# if user presses 's', the grayscale image is write to an image file
if key == ord("s"):
    
    cv2.imwrite('figs/vehicular-traffic-greyscale.png', img_out)
    print('output image has been saved in /figs/vehicular-traffic-greyscale.png')

# destroy windows to free memory  
cv2.destroyAllWindows()
print('windows have been closed properly - bye!')