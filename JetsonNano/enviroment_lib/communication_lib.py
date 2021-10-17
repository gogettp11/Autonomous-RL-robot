import serial

class Steer:
    def __init__(self):
        self.__serial = serial.Serial('/dev/ttyUSB0')
    def goRight(self):
        self.__serial.write(b'R')
    def goLeft(self):
        self.__serial.write(b'L')
    def goForward(self):
        self.__serial.write(b'F')

if __name__ == 'main':
    pass