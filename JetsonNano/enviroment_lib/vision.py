import cv2 as cv
import numpy as np

# lower boundary RED color range values; Hue (0 - 10)
lower1 = np.array([0, 100, 100])
upper1 = np.array([10, 255, 255])
 
# upper boundary RED color range values; Hue (160 - 180)
lower2 = np.array([160,100,100])
upper2 = np.array([179,255,255])

class Camera:
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

        return np.count_nonzero(frame_threshold > 250)

if __name__ == 'main':
    pass