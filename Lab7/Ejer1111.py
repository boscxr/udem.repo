# import required libraries
import numpy as np
import cv2
import math


# select a region of interest
def roi(img, vertices):
    """
    Applis a mask

    Only keeps the region of the image defined by the polygon
    formed from vertices. the rest is set to black
    """

    #starting with a blank mask
    mask = np.zeros_like(img)

    #define a 3 channel or 1 channel color to fill the mask
    #depending on the input image
    if len(img.shape)>2:
        channel_count = img.shape[2]
        ignore_mask = (255,)* channel_count
    else:
        ignore_mask = 255


    #fill pixels inside the polygon defined by vertices
    cv2.fillPoly(mask, vertices, ignore_mask)


    #returning the image only where mask pixels are nonzero
    imagewmask = cv2.bitwise_and(img, mask)
    return imagewmask

def processFrame(frame):

    #Convert RGB to greyscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Apply gaussian smoothing
    graussian = cv2.GaussianBlur(gray, (7, 7), 0)

    #Apply canny edge detector
    low_threshold = 10
    high_threshold = 70
    edges = cv2.Canny(graussian, low_threshold, high_threshold, apertureSize = 3)

    #Define a polygon-shape like region of interest
    img_shape = gray.shape
    
    img_size = img_shape

    # botleft = (0, img_size[0])
    # topleft = (0,0)
    # topright = (img_size[1],0)
    # botright = (img_size[1], img _size[0])

    #extract 
    botleft = (430, 840)
    topleft = (900, 580)
    topright = (1020, 580)
    botright = (1530, 838)

    #create an array that will ve used for the region of interest ROI
    vertices = np.array([[botleft,topleft,topright,botright]], dtype=np.int32)

    #get a region of interest ROI getting the polygon.
    #with hough transform we obtain estimated Hough Lines
    imagewmask = roi(edges,vertices)

    #Apply HOugh Transform for lane detection
    rho=1
    theta = np.pi/180
    threshold=40
    min_line_len=5
    max_line_gap=5
    #line_image= np.copy(img)*0  #create a blank to draw lines on
    hough_lines= cv2.HoughLinesP(imagewmask,rho,theta,threshold,np.array([]), 
        minLineLength=5, maxLineGap=5)

    #Visualise input and output images
    imagewmask = frame.copy()
    for line in hough_lines:
        for x1, y1, x2, y2 in line:
            cv2.line(imagewmask, (x1, y1), (x2,y2),(0,255,0),3)
   
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    for line in hough_lines:
        for x1,y1,x2,y2 in line:
            slope =(y2-y1)/(x2-x1)
            if math.fabs(slope) < 0.3:
                continue
            if slope<=0:
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:
                right_line_x.extend([x1,x2])
                right_line_y.extend([y1,y2])

    min_y = 850
    max_y = 600

    if left_line_x and left_line_y and right_line_x and right_line_y:


        img_colour_with_lines = frame.copy()

     

        poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x,deg=1))
        left_x_start= int(poly_left(max_y))
        left_x_end= int(poly_left(min_y))

        poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
        right_x_start= int(poly_right(max_y))
        right_x_end=int(poly_right(min_y))
        

    else:
        right_x_start = 1030
        right_x_end = 1580
        left_x_start = 890
        left_x_end = 570
        print(right_x_end)

    lineaslindas=[[[left_x_start, max_y, left_x_end, min_y],
    [right_x_start, max_y, right_x_end, min_y]]]



    img_colour_with_lines = frame.copy()
    for line in lineaslindas:
        for x1, y1, x2, y2 in line:
            cv2.line(img_colour_with_lines, (x1, y1), (x2, y2), (254,127,156), 10)


    #10.- Create polygon with points

    points = np.array([[[left_x_start+10,max_y],[right_x_start-10,max_y],
                                    [right_x_end-10,min_y],[left_x_end+10,min_y]]], np.int32)
                            
                        
    cv2.fillPoly(img_colour_with_lines, [points], (254, 55, 156))

    return img_colour_with_lines, imagewmask


# create a VideoCapture object and specify video file to be read
cap = cv2.VideoCapture("highway-right-solid-white-line-short.mp4")

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.namedWindow('MAgia', cv2.WINDOW_NORMAL)

# main loop
while(cap.isOpened()):

    # read current frame
    ret, frame = cap.read()

    # validate that frame was capture correctly
    if ret:
        
        output, output2 = processFrame(frame)

        # show current frame
        cv2.imshow('frame',output)
        cv2.imshow('MAgia',output2)

    # wait for the user to press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release VideoCapture object
cap.release()

# destroy windows to free memory
cv2.destroyAllWindows()