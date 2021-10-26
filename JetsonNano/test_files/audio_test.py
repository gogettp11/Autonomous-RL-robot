import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt 
from scipy import signal

#hyperparams
fs = 44100  # Sample rate
seconds = 1 # Duration of recording
sd.default.samplerate = fs
sd.default.channels = 1

while(True):
    print("start")
    myrecording = sd.rec(int(fs*seconds))
    sd.wait()  # Wait until recording is finished
    print("end")

    f_imag = np.fft.fft(np.squeeze(myrecording)).imag[950:1050]
    f_real = np.fft.fft(np.squeeze(myrecording)).real[950:1050]
    

    f_imag = signal.wiener(f_imag)
    f_real = signal.wiener(f_real)

    a = np.sqrt(np.square(f_real) + np.square(f_imag))
    print(max(a))