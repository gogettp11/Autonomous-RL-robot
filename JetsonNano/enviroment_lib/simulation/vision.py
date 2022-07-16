import cv2 as cv
import numpy as np
import rospy
# from ..abc_vision import Camera_abc
from std_msgs.msg import ByteMultiArray

# lower boundary RED color range values; Hue (0 - 10)
lower1 = np.array([0, 100, 100])
upper1 = np.array([10, 255, 255])
 
# upper boundary RED color range values; Hue (160 - 180)
lower2 = np.array([160,100,100])
upper2 = np.array([179,255,255])

CHUNKS = 5

class Camera_sim(object):
    def __init__(self):
        self.image = None
        rospy.init_node('camera_sim', anonymous=True)
        rospy.Subscriber('/camera1_ros/image_raw', ByteMultiArray, self.imageCallback)
    
    def imageCallback(self, data : ByteMultiArray, cb_args = None):
        self.image = data

    def getImageRedPixelsCount(self):

        while self.image is None:
            print("waiting for image...")
            rospy.sleep(1)
        
        img = self.image.deserialize_numpy()
        img = np.array(img.data)

        self.image = None

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

if __name__ == '__main__':
    # test of getImageRedPixelsCount()
    camera = Camera_sim()
    while True:
        print(camera.getImageRedPixelsCount())
        rospy.sleep(10)