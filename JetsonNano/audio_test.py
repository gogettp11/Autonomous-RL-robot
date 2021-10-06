import sounddevice as sd
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
from scipy import signal
import librosa
import librosa.display

#hyperparams
fs = 44100  # Sample rate
seconds = 1  # Duration of recording
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

mfccs = librosa.feature.mfcc(y=np.squeeze(myrecording,1), sr=fs, n_mfcc=40)
fig, ax = plt.subplots()
img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
ax.set(title='MFCC')
fig.colorbar(img, ax=ax)