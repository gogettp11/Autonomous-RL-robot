import gym
import rospy
import numpy as np

SOUND_FINISH_THRESHOLD = 1
VISION_REWARD_THRESHOLD = 3

class RealWorldEnv(gym.Env):
    def __init__(self, simulation = False):
        if simulation:
            from JetsonNano.enviroment_lib.simulation.steer import Steer_sim
            from JetsonNano.enviroment_lib.simulation.hearing import Microphone_sim
            from JetsonNano.enviroment_lib.simulation.vision import Camera_sim
            rospy.init_node('robot')
            self.__steer = Steer_sim()
            self.__camera = Camera_sim()
            self.__microphone = Microphone_sim()
        else:
            from JetsonNano.enviroment_lib.real_world.steer import Steer
            from JetsonNano.enviroment_lib.real_world.hearing import Microphone
            from JetsonNano.enviroment_lib.real_world.vision import Camera
            self.__steer = Steer()
            self.__camera = Camera()
            self.__microphone = Microphone()
        self.__movement_len = 500 # in miliseconds
        self.sound_before = 0
        self.cam_before = [0, 0, 0, 0, 0]

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
        
        cam_obs = self.__camera.getImageRedPixelsCount() # are there red objects in front of the robot? :[x,x,x,x,x]
        sound_amp = self.__microphone.returnFrequenciesMagnitudes() # how far is the sound from the microphone/goal position? :int

        for i in range(3): # camera observation space is 5 so middle has index 2
            if cam_obs[2-i] >= VISION_REWARD_THRESHOLD:
                reward -= 0.25*(3-i) # 3 to not multiply by 0
            if cam_obs[2+i] >= VISION_REWARD_THRESHOLD:
                reward -= 0.25*(3-i)
        
        if(sound_amp - self.sound_before >= 0):
            reward += 2.5
        else:
            reward -= 1

        if(sound_amp <= SOUND_FINISH_THRESHOLD):
            reward += 10
            done = True

        obs = np.concatenate([self.cam_before, cam_obs, np.expand_dims(sound_amp - self.sound_before , axis=0)])
        self.sound_before = sound_amp
        self.cam_before = cam_obs

        # print(f'obs: {obs} reward: {reward} done: {done}')

        return obs, done, reward, None
 
    def reset(self):
        # reset devices
        self.__steer.reset()
        self.__camera.reset()
        self.__microphone.reset()
        print("reset!")
        return np.zeros(11)
 
    def render(self, mode='human', close=False):
        pass

# if __name__ == '__main__':
#     # test enviroment with random agent
#     env = RealWorldEnv(simulation = True)
#     while True:
#         action = np.random.randint(0, 3)
#         obs, done, reward, info = env.step(action)
#         print(obs, done, reward, info)
#         if done:
#             break

# test reset
env = RealWorldEnv(simulation = True)
env.reset()