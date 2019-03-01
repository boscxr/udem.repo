# importa librerías estandar
import numpy as np
import cv2
import argparse

def options():
    # parse command line arguments
    parser = argparse.ArgumentParser('Read, visualise and write image into disk')
    parser.add_argument('-i', '--in_image_name', help='input image name', required=True)
    parser.add_argument('-o', '--out_image_name', help='output image name', required=True)
    args = vars(parser.parse_args())
    
    return args

def processing_image(img_in_name, img_out_name):
    
     # read in image from file
    img_in = cv2.imread(img_in_name, cv2.IMREAD_COLOR) # alternatively, you can use cv2.IMREAD_GRAYSCALE

    # verify that image exists
    if img_in is None:
        print('ERROR: image ', img_in_name, 'could not be read')
        exit()

    # convert input image from colour to grayscale
    img_out = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

    # create a new window for image purposes
    cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)  # alternatively, you can use cv2.WINDOW_NORMAL
    cv2.namedWindow("output image", cv2.WINDOW_AUTOSIZE) # that option will allow you for window resizing

    # visualise input and output image
    cv2.imshow("input image", img_in)
    cv2.imshow("output image", img_out)

    # wait for the user to press a key
    key = cv2.waitKey(0)

    # if user pressed 's', the grayscale image is write to disk
    if key == ord("s"):
        cv2.imwrite(img_out_name, img_out)
        print('output image has been saved in ../figs/vehicular-traffic-greyscale.png')

    # destroy windows to free memory  
    cv2.destroyAllWindows()
    print('windows have been closed properly')

    
# main function
def main():    
    
    # uncomment these lines when running on jupyter notebook
    # and comment when running as a script on linux terminal
    args = {
            "in_image_name": "figs/vehicular-traffic.jpg",
            "out_image_name": "figs/vehicular-traffic-greyscale.png"
            }
    
    # comment the following line when running on jupyter notebook
    # and uncomment when running as a script on linux terminarl
    #args = options()
    
    in_image_name = args['in_image_name']
    out_image_name = args['out_image_name']
    
    # call processing image
    processing_image(in_image_name, out_image_name)
    
    
# run first
if __name__=='__main__':
    main()