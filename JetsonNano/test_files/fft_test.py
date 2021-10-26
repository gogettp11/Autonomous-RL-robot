from numpy import fft
a = [0,0,1,0]
n = fft.fft(a)
print(n)
b = fft.rfft(a)
print(b)
# https://www.mathworks.com/matlabcentral/answers/225440-how-to-extract-phase-and-amplitude-information-from-fft

aa = b'L'
bb = aa + str(1000).encode()

print(bb)

print(3//2)