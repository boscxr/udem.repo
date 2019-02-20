# import required libraries
import matplotlib.pyplot as plt
import cv2

# print image statistics
def print_image_statistics(img, head_string):

    """
        In this def the statistics of every image
        is printed in the terminal
    """
    img_size=img.shape

    print('\n'+head_string)
    print('image size: ', img_size)

    # retrieve image width resolution
    print('image width resolution: ', img_size[0])

    # retrieve image height resolution
    print('image height resolution: ', img_size[1])

    # retrieve number of channels
    if img is img_colour:
        print('number of channels: ', img_size[2])
    else:
        print('number of channels: ', 1)
    # minimum pixel value in image
    print('minimum intensity value: ', img.min())

    # minimum pixel value in image
    print('max intensity value: ', img.max())

    # maximum intensity value in image
    print('meam intensity value: ', img.mean())

    # print type of image
    print('type of image: ', img.dtype)

    return None


# visualise image
def visualise_image(img, fig_number, fig_title, iscolour):

    """
        We can visualise the images in this def
    """
    plt.figure(fig_number)
    if iscolour == 1:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray') 
    plt.title(fig_title)
    plt.xlabel('x-resolution')
    plt.ylabel('y-resolution')

    return None

# read image
image_name = 'figs/vehicular_traffic.jpg'
img_colour = cv2.imread(image_name, cv2.IMREAD_COLOR)
img_colour = cv2.cvtColor(img_colour, cv2.COLOR_BGR2RGB)

# verify that image exists
if img_colour is None:
    print('ERROR: image ', image_name, 'could not be read')
    exit()

# convert the input colour image into a grayscale image
img_greyscale = cv2.cvtColor(img_colour, cv2.COLOR_BGR2GRAY)

# print colour image stats and visualise it
print_image_statistics(img_colour, 'COLOUR IMAGE STATS:')
visualise_image(img_colour, 1, 'INPUT IMAGE: COLOUR', 1)

# print greyscale image stats and visualise it
print_image_statistics(img_greyscale, 'GREYSCALE IMAGE STATS:')
visualise_image(img_greyscale, 2, 'OUTPUT IMAGE: GREYSCALE', 0)

# visualise figures
plt.show()