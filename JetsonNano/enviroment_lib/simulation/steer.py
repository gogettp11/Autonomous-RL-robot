import rospy
from std_msgs.msg import ByteMultiArray
# from ..abc_steer import Steer_abc

class Steer_sim(object):
    def __init__(self):
        rospy.init_node('steer_sim')
        self.__pub = rospy.Publisher('/vel_cmd', ByteMultiArray, queue_size=1)

    def goRight(self, lenght : int):
        self.__pub.publish(ByteMultiArray(data=[0x40, 0x0]))
        rospy.sleep(lenght)
        return True

    def goLeft(self, lenght : int):
        self.__pub.publish(ByteMultiArray(data=[0x0, 0x40]))
        rospy.sleep(lenght)
        return True

    def goForward(self, lenght : int):
        self.__pub.publish(ByteMultiArray(data=[0x40, 0x40]))
        rospy.sleep(lenght)
        return True

if __name__ == '__main__':
    # test for steering
    steer = Steer_sim()
    # sequence of commands in loop
    while True:
        steer.goForward(1)
        steer.goLeft(1)
        steer.goRight(1)