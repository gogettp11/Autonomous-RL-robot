import sounddevice as sd
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

#hyperparams
fs = 48000  # Sample rate
seconds = 2  # Duration of recording
sd.default.samplerate = fs
sd.default.channels = 1

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

print("start")
myrecording = sd.rec(int(fs*seconds))
sd.wait()  # Wait until recording is finished
print("end")

myrecording = butter_highpass()
fft = np.fft.fft(myrecording)
plt.plot(fft)
plt.show()
