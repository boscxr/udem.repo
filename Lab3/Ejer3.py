# import required libraries
import matplotlib.pyplot as plt
import cv2

# read image
image_name = 'figs/vehicular_traffic.jpg'
img_colour = cv2.imread(image_name, cv2.IMREAD_COLOR)
img_colour = cv2.cvtColor(img_colour, cv2.COLOR_BGR2RGB)

# verify that image exists
if img_colour is None:
    print('ERROR: image ', image_name, 'could not be read')
    exit()


# visualise image using matplotlib
plt.figure(1)
plt.imshow(img_colour)
plt.title('INPUT IMAGE: COLOUR')
plt.xlabel('x-resolution')
plt.ylabel('y-resolution')


# create a region of interest
# NOTE THE FOLLOWING:
# FIRST RANGE REPRESENTS COORDINATES FOR THE Y-AXIS
# SECOND RANGE REPRESENTS COORDINATES FOR THE X-AXIS
img_roi = img_colour[550:700, 630:750]

# visualise region of interest ROI in a new figure
plt.figure(2)
plt.imshow(img_roi)
plt.title('OUTPUT IMAGE: Region Of Interest')
plt.xlabel('x-resolution')
plt.ylabel('y-resolution')

# overlay roi on colour image
img_colour[570:720, 880:1000] = img_roi
#img_colour[550:700, 450:570] = img_roi

# visualise region of interest ROI on colour image
plt.figure(3)
plt.imshow(img_colour)
plt.title('roi image overlaid on colour image')
plt.xlabel('x-resolution')
plt.ylabel('y-resolution')

# display figures
plt.show()