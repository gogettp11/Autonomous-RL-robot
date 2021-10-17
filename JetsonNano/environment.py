import gym
from gym import error, spaces, utils
from gym.utils import seeding

from enviroment_lib.communication_lib import Steer
from enviroment_lib.hearing import Microphone
from enviroment_lib.vision import Camera
 
class RealWorldEnv(gym.Env):
    def __init__(self):
        self.__steer = Steer()
        self.__camera = Camera()
        self.__microphone = Microphone()

    def step(self, action): #return new observations
        pass
 
    def reset(self): # xD
        print("you have to do it by yourself")
 
    def render(self, mode='human', close=False):
        pass

if __name__ == 'main':
    pass