import rospy
import numpy as np
from gazebo_msgs.msg import ModelStates
import threading

class Microphone_sim(object):
    def __init__(self):
        # robot position subscriber
        rospy.init_node('microphone_sim')
        self.__sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.__callback)
        self.robot_name = 'TinyBot_camera_friction_test'
        self.current_position = None
        self.lock = threading.Lock()
        self.goal_position = (10, 10)

        while self.current_position is None:
            rospy.sleep(0.1)
        
        self.index = self.current_position.name.index(self.robot_name)

    def __callback(self, data):
        with self.lock:
            self.current_position = data
    
    def returnFrequenciesMagnitudes(self):
        
        while self.current_position is None:
            rospy.sleep(0.1)
        
        with self.lock:
            current_position = self.current_position
            self.current_position = None
        pos =  current_position.pose[self.index].position
        
        x = pos.x
        y = pos.y

        # calculate distance between current position and goal position
        distance = np.round(np.sqrt((x - self.goal_position[0])**2 + (y - self.goal_position[1])**2))
        return distance


if __name__ == '__main__':
    # test for microphone
    microphone = Microphone_sim()
    # check if the robot is in the right position
    while True:
        print(microphone.returnFrequenciesMagnitudes())
        rospy.sleep(0.2)