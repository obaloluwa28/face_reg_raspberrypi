import cv2
import os
import json
import time
import numpy as np
from PIL import Image
import pickle
import subprocess
import RPi.GPIO as GPIO
import serial
import sys
from signal import signal, SIGTERM, SIGHUP
from rpi_lcd import LCD
lcd = LCD()

def safe_exit(signum, frame):
    exit(1)

# Declear Serial Communication
ser = serial.Serial(
        port='/dev/ttyS0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
        baudrate = 115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=None
)

# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier(
    '/home/pi/Desktop/facerec/Newfolder/OpenCV-Face-Recognition-Python/src/cascades/data/haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("/home/pi/Desktop/facerec/Newfolder/OpenCV-Face-Recognition-Python/src/recognizers/face-trainner.yml")

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'src/cascades/data/haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier('/home/pi/Desktop/facerec/Newfolder/OpenCV-Face-Recognition-Python/src/cascades/data/haarcascade_frontalface_alt.xml')

labels = {"person_name": 1}
with open("/home/pi/Desktop/facerec/Newfolder/OpenCV-Face-Recognition-Python/src/pickles/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

# For each person, one face id
face_id = 9

# Initialize sample face image
count = 0

# Parent Directory path
parent_dir = "/home/pi/Desktop/facerec/Newfolder/OpenCV-Face-Recognition-Python/src/images/"

# def Trainface
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

def ScanOldFaces():
    global count
    vid_cam = cv2.VideoCapture(0)
    while (count<=3):
        lcd.text("Scanning... "+ str(count) , 1)
        lcd.text("Place Your Face", 2)
        # Capture frame-by-frame
        _, frame =vid_cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]  # (ycord_start, ycord_end)
            roi_color = frame[y:y+h, x:x+w]
            # recognize? deep learned model predict keras tensorflow pytorch scikit learn
            
            id_, conf = recognizer.predict(roi_gray)
            conf=100-float(conf)
            print (conf, "%")
            if conf >= 4 and conf <= 95:
                print("id:", id_)
                print(labels[id_])
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                txtcolor = (255, 255, 255)
                stroke = 2
                cv2.putText(frame, name, (x, y), font, 1, txtcolor, stroke, cv2.LINE_AA)
            else:
                name = "Unknown"
            framecolor = (255, 0, 0)  # BGR 0-255
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), framecolor, stroke)
            count += 1
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        if (count > 3):
            if(name!="Unknown"):
                ser.write(name.encode())
                lcd.text("Attendance Marked", 1)
                lcd.text(str(name)+"  Present", 2)
                name = ''
                vid_cam.release()
            else:
                lcd.clear()
                lcd.text("Unknown Face", 1)
                vid_cam.release()
    return


while True:
    #This Line of code Sends attendance taken to the ESP then to the cloud
    ScanOldFaces()
    if count > 3:
        count = 0
        cv2.destroyAllWindows()
        break