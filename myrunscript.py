import subprocess
import time
import os
import RPi.GPIO as GPIO
from signal import signal, SIGTERM, SIGHUP, pause
from rpi_lcd import LCD
lcd = LCD()


def safe_exit(signum, frame):
    exit(1)


signal(SIGTERM, safe_exit)
signal(SIGHUP, safe_exit)

GPIO.setmode(GPIO.BCM)
GPIO.setup(24, GPIO.IN)


while True:
    input_state1 = GPIO.input(24)
    if input_state1 == False:
        time.sleep(1)
        p.terminate()
        cmd2 = 'python3 /home/pi/Desktop/facerec/Newfolder/OpenCV-Face-Recognition-Python/src/registerface.py'
        pol = subprocess.Popen(
            "exec " + cmd2, stdout=subprocess.PIPE, shell=True)
        pol.wait()
    else:
        cmd = 'python3 /home/pi/Desktop/facerec/Newfolder/OpenCV-Face-Recognition-Python/src/face_datasets.py'
        p = subprocess.Popen("exec " + cmd, stdout=subprocess.PIPE, shell=True)
        p.wait()
