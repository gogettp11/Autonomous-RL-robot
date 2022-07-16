import gym

#real
from JetsonNano.enviroment_lib.real_world.steer import Steer
from JetsonNano.enviroment_lib.real_world.hearing import Microphone
from JetsonNano.enviroment_lib.real_world.vision import Camera
#simulation
from JetsonNano.enviroment_lib.simulation.steer import Steer_sim
from JetsonNano.enviroment_lib.simulation.hearing import Microphone_sim
from JetsonNano.enviroment_lib.simulation.vision import Camera_sim

SOUND_REWARD_THRESHOLD = 80
SOUND_FINISH_THRESHOLD = 2000
VISION_REWARD_THRESHOLD = 4

class RealWorldEnv(gym.Env):
    def __init__(self, simulation = False):
        if simulation:
            self.__steer = Steer_sim()
            self.__camera = Camera_sim()
            self.__microphone = Microphone_sim()
        else:
            self.__steer = Steer()
            self.__camera = Camera()
            self.__microphone = Microphone()
        self.__movement_len = 1000 # in miliseconds
        self.sound_before = 0

    # return new_observations, reward, is_done
    def step(self, action): # action space = {0,1,2}

        result = False
        reward = 0
        done = False

        if(action == 0): # forward
            result = self.__steer.goLeft(self.__movement_len)
        elif(action == 1): # left
            result = self.__steer.goForward(self.__movement_len)
        elif(action == 2): # right
            result = self.__steer.goRight(self.__movement_len)

        if not result:
            return None, None, None, None
        
        cam_obs = self.__camera.getImageRedPixelsCount()
        sound_amp = self.__microphone.returnFrequenciesMagnitudes() # stop for one sec and listen

        if(cam_obs[len(cam_obs//2)] >= VISION_REWARD_THRESHOLD):
            reward -= 2
        
        if(sound_amp - self.sound_before > SOUND_REWARD_THRESHOLD):
            reward += 1

        if(sound_amp > SOUND_FINISH_THRESHOLD):
            done = True

        return cam_obs, done, reward, None
 
    def reset(self):
        print("you have to do it by yourself")
 
    def render(self, mode='human', close=False):
        pass

if __name__ == 'main':
    pass