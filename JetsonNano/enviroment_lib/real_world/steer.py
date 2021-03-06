import serial
# from ..abc_steer import Steer_abc

class Steer(object):
    def __init__(self):
        self.__serial = serial.Serial('/dev/ttyUSB0')
    def goRight(self, lenght : int):
        self.__serial.write(b'R' + str(lenght).encode())
        data = self.__serial.read(1)
        if(data.decode() != 'R'):
            return False
        return True

    def goLeft(self, lenght : int):
        self.__serial.write(b'L' + str(lenght).encode())
        data = self.__serial.read(1) 
        if(data.decode() != 'L'):
            return False
        return True

    def goForward(self, lenght : int):
        self.__serial.write(b'F' + str(lenght).encode())
        data = self.__serial.read(1)
        if(data.decode() != 'F'):
            return False
        return True

if __name__ == '__main__':
    # test 
    steer = Steer()
    steer.goRight(300)
    steer.goLeft(300)
    steer.goForward(300)