import cv2 as cv
import numpy as np
import rospy
from ..abc_vision import Camera_abc

# lower boundary RED color range values; Hue (0 - 10)
lower1 = np.array([0, 100, 100])
upper1 = np.array([10, 255, 255])
 
# upper boundary RED color range values; Hue (160 - 180)
lower2 = np.array([160,100,100])
upper2 = np.array([179,255,255])

CHUNKS = 5

class Camera_sim(Camera_abc):
    def __init__(self):
        self.__camera = cv.VideoCapture(0)

    def getImageRedPixelsCount(self):
        ret,img = self.__camera.read()

        if img is None:
            return "error"

        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        lower_mask = cv.inRange(img_hsv, lower1, upper1)
        upper_mask = cv.inRange(img_hsv, lower2, upper2)
        
        frame_threshold = lower_mask + upper_mask
        frame_threshold = cv.GaussianBlur(frame_threshold, (13,13), 5,None,5)

        temp_array = []

        width = img.shape[1] # width
        chunk_size = width//CHUNKS
        chunk_start = 0
        chunk_end = chunk_size
        for i in range(CHUNKS):
            temp_array.append(np.count_nonzero(frame_threshold[:,chunk_start:chunk_end] > 250)//1500)
            chunk_start = chunk_end
            chunk_end += chunk_size

        return temp_array

if __name__ == 'main':
    pass