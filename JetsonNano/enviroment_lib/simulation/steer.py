import rospy
import time
import numpy as np
from std_msgs.msg import ByteMultiArray
from std_srvs.srv import Empty
# from ..abc_steer import Steer_abc

class Steer_sim(object):
    def __init__(self):
        self.__pub = rospy.Publisher('/vel_cmd', ByteMultiArray, queue_size=1)
        self.reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

    def goRight(self, lenght : int):
        lenght = np.divide(lenght, 1000) # ms to s
        self.__pub.publish(ByteMultiArray(data=[0x25, 0x0]))
        time.sleep(lenght)
        return True

    def goLeft(self, lenght : int):
        lenght = np.divide(lenght, 1000) # ms to s
        self.__pub.publish(ByteMultiArray(data=[0x0, 0x25]))
        time.sleep(lenght)
        return True

    def goForward(self, lenght : int):
        lenght = np.divide(lenght, 1000) # ms to s
        self.__pub.publish(ByteMultiArray(data=[0x40, 0x40]))
        time.sleep(lenght)
        return True
    
    # reset the robot position
    def reset(self):
        self.reset_simulation()
        return True


if __name__ == '__main__':
    # test for steering
    steer = Steer_sim()
    # sequence of commands in loop
    while True:
        steer.goForward(1)
        steer.goLeft(1)
        steer.goRight(1)
