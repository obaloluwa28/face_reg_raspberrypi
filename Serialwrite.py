#!/usr/bin/env python
import time
import serial

ser = serial.Serial(
        port='/dev/ttyUSB0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
        baudrate = 115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=None
)
matric=""

while True:
#        This Line of code Sends attendance taken to the ESP then to the cloud
        if(matric != ''):
            ser.write(matric.encode())
            matric = ''
            time.sleep(0.5)
        
#        read input from the serial monitor
        newMatric=ser.readline()
        print("New Matric:",newMatric.decode('utf-8').rstrip())
        time.sleep(0.5)
        
        
        