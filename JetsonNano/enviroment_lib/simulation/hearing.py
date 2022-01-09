import rospy
import numpy as np
from ..abc_hearing import Microphone_abc

class Microphone_sim(Microphone_abc):
    def __init__(self):
        pass
    
    def returnFrequenciesMagnitudes(self):
        myrecording = sd.rec(int(self.__sample_rate*self.__seconds))
        sd.wait()  # Wait until recording is finished

        f_imag = np.fft.fft(np.squeeze(myrecording)).imag
        f_real = np.fft.fft(np.squeeze(myrecording)).real

        return max(np.sqrt(np.square(f_real[900:1100]) + np.square(f_imag[900:1100]))) # about 1000hz

if __name__ == 'main':
    pass