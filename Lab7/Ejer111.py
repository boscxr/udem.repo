"""
    object-segmentation-using-hsv-colour-space.py

    add a description of your code here

    author: add your fullname
    date created: add this info
    universidad de monterrey
"""

# import required libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2



# select a region of interest
def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# run line detection pipeline
def run_pipeline(img_name):

    # 1.- Read image
    img_colour = cv2.imread(img_name)

    # verify that image `img` exist
    if img_colour is None:
        print('ERROR: image ', img_name, 'could not be read')
        exit()

    # 2. Convert from BGR to RGB then from RGB to greyscale
    img_colour_rgb = cv2.cvtColor(img_colour, cv2.COLOR_BGR2RGB)
    grey = cv2.cvtColor(img_colour_rgb, cv2.COLOR_RGB2GRAY)

    # 3.- Apply Gaussuan smoothing
    kernel_size = (7,7)
    blur_grey = cv2.GaussianBlur(grey, kernel_size, sigmaX=0, sigmaY=0)

    # 4.- Apply Canny edge detector
    low_threshold = 10
    high_threshold = 70
    edges = cv2.Canny(blur_grey, low_threshold, high_threshold, apertureSize=3)

    # 5.- Define a polygon-shape like region of interest
    img_shape = grey.shape

    # uncomment the following lines when extracting lines around the whole image
    
    # img_size = img_shape
    # bottom_left = (0, img_size[0])
    # top_left = (0, 0)
    # top_right = (img_size[1], 0)
    # bottom_right = (img_size[1], img_size[0])
    

    # comment the following lines when extracting  lines around the whole image

    bottom_left = (388, 626)
    top_left = (600, 460)
    top_right = (777, 458)
    bottom_right = (1128, 634)

    # create a vertices array that will be used for the roi
    vertices = np.array([[bottom_left,top_left, top_right, bottom_right]], dtype=np.int32)

    # 6.- Get a region of interest using the just created polygon. This will be
    #     used together with the Hough transform to obtain the estimated Hough lines
    masked_edges = region_of_interest(edges, vertices)

    # 7.- Apply Hough transform for lane lines detection
    rho = 1                       # distance resolution in pixels of the Hough grid
    theta = np.pi/180             # angular resolution in radians of the Hough grid
    threshold = 40                # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 5              # minimum number of pixels making up a line
    max_line_gap = 5              # maximum gap in pixels between connectable line segments
    line_image = np.copy(img_colour)*0   # creating a blank to draw lines on
    hough_lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    print(hough_lines)


    # 8.- Visualise input and output images he is putting all lines in matplot
    # img_colour_with_lines = img_colour_rgb.copy()
    # for line in hough_lines:
    #     for x1, y1, x2, y2 in line:
    #         cv2.line(img_colour_with_lines, (x1, y1), (x2, y2), (0,255,0), 3)

    #9.- Extend Hough Dominant Lines

    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    for line in hough_lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.
            if math.fabs(slope) < 0.5: # <-- Only consider extreme slope
                continue
            if slope <= 0: # <-- If the slope is negative, left group.
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else: # <-- Otherwise, right group.
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])


    #Limites superior e inferior
    min_y = 635 # <-- Just below the horizon
    max_y = 450 # <-- The bottom of the image


    #Aqui se generan las lineas 
    poly_left = np.poly1d(np.polyfit(
    left_line_y,
    left_line_x,
    deg=1))

    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))

    poly_right = np.poly1d(np.polyfit(
    right_line_y,
    right_line_x,
    deg=1))

    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))

    #Guardo en una variable las lineas
    lineaslindas= [[[left_x_start, max_y, left_x_end, min_y],
        [right_x_start, max_y, right_x_end, min_y]]]


    img_colour_with_lines = img_colour_rgb.copy()
    for line in lineaslindas:
        for x1, y1, x2, y2 in line:
            cv2.line(img_colour_with_lines, (x1, y1), (x2, y2), (254,127,156), 10)


    #10.- Create polygon with points

    points = np.array([[[left_x_start+10,max_y],[right_x_start-10,max_y],
                                    [right_x_end-10,min_y],[left_x_end+10,min_y]]], np.int32)
                            
                        
    cv2.fillPoly(img_colour_with_lines, [points], (254, 55, 156))

    #visualise input and output images
    # plt.figure(1)
    # plt.imshow(img_colour_rgb)
    # plt.axis('off')

    # plt.figure(2)
    # plt.imshow(blur_grey, cmap='gray')
    # plt.axis('off')

    # plt.figure(3)
    # plt.imshow(edges, cmap='gray')
    # plt.axis('off')


    # create a VideoCapture object

    return img_colour_with_lines

#################################################################################


# config video object
def config_video_source(source_video_file):

    # initialise a video capture object
    cap = cv2.VideoCapture(source_video_file)

    # check that the videocapture object was successfully created
    if(not cap.isOpened()):
        print("Error opening video source")
        exit()

    # return videocapture object
    return cap


# process video
def process_video(cap):


    # create new windows for visualisation purposes
    cv2.namedWindow('input video', cv2.WINDOW_NORMAL)
    cv2.namedWindow('segmented object', cv2.WINDOW_NORMAL)

    # main loop
    while(cap.isOpened()):

        # grab current frame
        ret, frame = cap.read()

        # verify that frame was properly captured
        if ret == False:
            print("ERROR: current frame could not be read")            
            break

        # convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # threshold the hsv image so that only blue pixels are kept
        #mask = cv2.inRange(hsv, hsv_min, hsv_max)



        # AND-bitwise operation between the mask and input images
        #segmented_objects = cv2.bitwise_and(frame, frame, mask=mask)

        segmented_objects = run_pipeline(hsv)

        # visualise current frame
        cv2.imshow('input video',frame)

        # visualise segmented blue object
        cv2.imshow('segmented object', segmented_objects)

        # Display the resulting frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('volti-swimming.png', frame)
            cv2.imwrite('volti-swimming-segmented.png', segmented_objects)
            break


# free memory and close open windows
def free_memory_amd_close_windows(cap):

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


# run pipeline
def run_pipeline(dataset_number=1):


    # select video sequence
    if(dataset_number==1):
        # blue
        cap = config_video_source(source_video_file="highway-right-solid-white-line-short.mp4")
        process_video(cap)
    

    else:
        # enter a valid option
        print("Please, enter a valid option mate! [1-4]")
        exit()

    # free memory and close windows
    free_memory_amd_close_windows(cap)


# main function
def main():    
    run_pipeline(dataset_number = 1)


if __name__=='__main__':
    main()
