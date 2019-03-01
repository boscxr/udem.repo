# import required libraries
import numpy as np
import cv2

# create a VideoCapture object and specify video file to be read
cap = cv2.VideoCapture('figs/highway_right_solid_white_line_short.mp4')

# main loop
while(cap.isOpened()):

    # read current frame
    ret, frame = cap.read()

    # validate that frame was capture correctly
    if ret:
        
        # convert frame from colour to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # show current frame
        cv2.imshow('frame',gray)

    # wait for the user to press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release VideoCapture object
cap.release()

# destroy windows to free memory
cv2.destroyAllWindows()