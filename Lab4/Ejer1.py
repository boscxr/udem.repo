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
def process_video(cap, hsv_min=(20, 20, 29), hsv_max=(40, 200, 200)):


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
        mask = cv2.inRange(hsv, hsv_min, hsv_max)

        # AND-bitwise operation between the mask and input images
        segmented_objects = cv2.bitwise_and(frame, frame, mask=mask)

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
        cap = config_video_source(source_video_file="mardigrass.mp4")
        process_video(cap, hsv_min=(95, 40, 100), hsv_max=(110, 255, 255))
        # red 
        cap = config_video_source(source_video_file="mardigrass.mp4")
        process_video(cap, hsv_min=(160, 65, 85), hsv_max=(180, 200, 200))
        # yellow 
        cap = config_video_source(source_video_file="mardigrass.mp4")
        process_video(cap, hsv_min=(17, 40, 100), hsv_max=(32, 255, 255))
        # violet
        cap = config_video_source(source_video_file="mardigrass.mp4")
        process_video(cap, hsv_min=(125, 40, 100), hsv_max=(140, 255, 255))
                                                           
    elif(dataset_number==2):
        # blue ish
        cap = config_video_source(source_video_file="volti-01.mp4")
        process_video(cap, hsv_min=(115, 30, 0), hsv_max=(190, 250, 250))
        #red ish 
        cap = config_video_source(source_video_file="volti-01.mp4")
        process_video(cap, hsv_min=(93, 30, 0), hsv_max=(108, 250, 250))
        

    else:
        # enter a valid option
        print("Please, enter a valid option mate! [1-4]")
        exit()

    # free memory and close windows
    free_memory_amd_close_windows(cap)


# main function
def main():    
    run_pipeline(dataset_number = 2)


if __name__=='__main__':
    main()