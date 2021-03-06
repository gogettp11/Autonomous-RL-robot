import sounddevice as sd
import time
import numpy as np
# from ..abc_hearing import Microphone_abc

class Microphone(object):
    def __init__(self):
        self.__sample_rate = 44100
        self.__seconds = 1
        sd.default.samplerate = self.__sample_rate
        sd.default.channels = 1
        self.goal_amplitude = 2000
    
    def returnFrequenciesMagnitudes(self):
        myrecording = sd.rec(int(self.__sample_rate*self.__seconds))
        sd.wait()  # Wait until recording is finished

        f_imag = np.fft.fft(np.squeeze(myrecording)).imag
        f_real = np.fft.fft(np.squeeze(myrecording)).real

        return (self.goal_amplitude - max(np.sqrt(np.square(f_real[900:1100]) + np.square(f_imag[900:1100]))))//100 # about 1000hz

if __name__ == '__main__':
    # test for microphone
    microphone = Microphone()
    # reading loop
    while True:
        print(microphone.returnFrequenciesMagnitudes())
        time.sleep(0.2)