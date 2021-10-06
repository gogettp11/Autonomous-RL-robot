import serial

ser = serial.Serial('/dev/ttyUSB0')
ser.write(b'r')