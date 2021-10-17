import sounddevice as sd
import numpy as np

class Microphone:
    def __init__(self):
        self.__sample_rate = 44100
        self.__seconds = 1
        sd.default.samplerate = self.__sample_rate
        sd.default.channels = 1
    
    def returnFrequenciesMagnitudes(self):
        myrecording = sd.rec(int(self.__sample_rate*self.__seconds))
        sd.wait()  # Wait until recording is finished

        f_imag = np.fft.fft(np.squeeze(myrecording)).imag
        f_real = np.fft.fft(np.squeeze(myrecording)).real

        return np.sqrt(np.square(f_real[:2000]) + np.square(f_imag[:2000])) # up to 2000hz

if __name__ == 'main':
    pass