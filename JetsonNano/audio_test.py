import sounddevice as sd
from matplotlib import pyplot as plt

fs = 44100  # Sample rate
seconds = 2  # Duration of recording

sd.default.samplerate = fs
sd.default.channels = 1

# print(sd.get_status())

myrecording = sd.rec(int(fs*seconds))
sd.wait()  # Wait until recording is finished

plt.plot(myrecording)
plt.show()
