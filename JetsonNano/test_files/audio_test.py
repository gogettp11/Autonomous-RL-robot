import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt 

#hyperparams
fs = 44100  # Sample rate
seconds = 1 # Duration of recording
sd.default.samplerate = fs
sd.default.channels = 1

print("start")
myrecording = sd.rec(int(fs*seconds))
sd.wait()  # Wait until recording is finished
print("end")

f_imag = np.fft.fft(np.squeeze(myrecording)).imag
f_real = np.fft.fft(np.squeeze(myrecording)).real

plt.plot(np.sqrt(np.square(f_real) + np.square(f_imag)))
plt.show()