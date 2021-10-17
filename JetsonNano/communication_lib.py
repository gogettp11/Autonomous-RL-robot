import serial

class Steer:
    self.__serial = serial.Serial('/dev/ttyUSB0')

    def goRight(self):
        self.__serial.write(b'R')

# ser.write(b'r')