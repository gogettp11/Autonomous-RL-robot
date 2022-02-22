import rospy 
import rospkg 
from gazebo_msgs.srv import ApplyBodyWrench
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
from gazebo_msgs.msg import *
from geometry_msgs.msg import Wrench
from sensor_msgs.msg import Image
from time import sleep
import numpy as np
import cv2 as cv
from copy import deepcopy

DATA = None

def cb(data):
    global DATA
    decoded = np.frombuffer(data.data, dtype=np.uint8)
    DATA = deepcopy(decoded.reshape((240, 320,3)))
    print("got image")

class Point:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
def main():
    rospy.init_node('set_pose')

    state_msg = Wrench()
    model = GetModelStateRequest()
    model.model_name='TinyBot_camera_friction'

    image = rospy.Subscriber("/camera1_ros/image_raw", Image, cb)

    state_msg.torque.x = -5
    
    rospy.wait_for_service('/gazebo/apply_body_wrench')
    set_state = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)

    rospy.wait_for_service ('/gazebo/get_model_state')
    get_model_srv = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

    pos = get_model_srv(model)
    x = pos.pose.orientation.x
    y = pos.pose.orientation.y
    z = pos.pose.orientation.y
    w = pos.pose.orientation.w
    print(f'x: {x} y: {y} z: {z} w:{w}')


    resp = set_state(wrench=state_msg, reference_frame = 'TinyBot_camera_friction::left_wheel',
                    duration = rospy.Duration(2), body_name = 'TinyBot_camera_friction::left_wheel')
    resp = set_state(wrench=state_msg, reference_frame = 'TinyBot_camera_friction::left_wheel',
                    duration = rospy.Duration(2), body_name = 'TinyBot_camera_friction::right_wheel')
    print(resp)

    # cv.imshow('title', DATA)
    # cv.waitKey(0)
  
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass