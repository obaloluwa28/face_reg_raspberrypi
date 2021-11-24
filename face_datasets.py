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

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(18,GPIO.OUT)

# Declear Serial Communication
ser = serial.Serial(
        port='/dev/ttyUSB0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
        baudrate = 115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=None
)

# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier(
    '/home/pi/Desktop/face rec/New folder/OpenCV-Face-Recognition-Python/src/cascades/data/haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("/home/pi/Desktop/face rec/New folder/OpenCV-Face-Recognition-Python/src/recognizers/face-trainner.yml")

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'src/cascades/data/haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier('/home/pi/Desktop/face rec/New folder/OpenCV-Face-Recognition-Python/src/cascades/data/haarcascade_frontalface_alt.xml')

labels = {"person_name": 1}
with open("/home/pi/Desktop/face rec/New folder/OpenCV-Face-Recognition-Python/src/pickles/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

# For each person, one face id
face_id = 9

# Initialize sample face image
count = 0
county = 0

# Start capturing video
#vid_cam = cv2.VideoCapture(0)

# Parent Directory path
parent_dir = "/home/pi/Desktop/face rec/New folder/OpenCV-Face-Recognition-Python/src/images/"

# def Trainface
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")


# define Buzzer function
def Buzz():
    for x in range(3):
        # GPIO.output(18,GPIO.HIGH)
        print("Make Sound")
        time.sleep(1)
        # GPIO.output(18,GPIO.LOW)
        print("Keep Quiet")
        time.sleep(1)
    return


# Function Trains Face Model All Over After It is being captured
def Trainface():
    lcd.text("Training Model", 1)
    current_id = 1
    label_ids = {}
    y_labels = []
    x_train = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
#                print(label, path)
                # if the folder name(Label) doesnet already exist in the Label_id dictionary
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]
                print("Label:", label)

                pil_image = Image.open(path).convert("L")  # grayscale
                image_array = np.array(pil_image, "uint8")
                faces = face_detector.detectMultiScale(image_array, 1.3, 5)

                for (x, y, w, h) in faces:
                    roi = image_array[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)

    with open("/home/pi/Desktop/face rec/New folder/OpenCV-Face-Recognition-Python/src/pickles/face-labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("/home/pi/Desktop/face rec/New folder/OpenCV-Face-Recognition-Python/src/recognizers/face-trainner.yml")
    return

# Function Captures new face and saves it in the required folder
def captureNewface(matric_num):
    # Value returned from the ESP-01
    global county
    vid_cam = cv2.VideoCapture(0)
    input_string = matric_num
    print(input_string)
    matric_num = input_string[0:6]
    
    if (matric_num != ""):
        # Display on LCD "Enrolling New Face"
        signal(SIGTERM, safe_exit)
        signal(SIGHUP, safe_exit)
        lcd.text("Enroll New Face", 1)
        lcd.text("Initializing...", 2)
        # Buzzer Sounds 5times at {500ms interval}
        Buzz()
        # Display on LCD "Place Image Before The Camera"
        lcd.clear()
        lcd.text("Scanning...", 1)
        lcd.text("Place Your Face", 2)

        # def create_dir(matric_num):
        if not os.path.isdir(matric_num):
            # Create a folder with the student's Matric Number
            path = os.path.join(parent_dir, matric_num)
            os.mkdir(path)
            print(matric_num, " created Successfully!")
        else:
            print("Folder Already Existed")

        # Change your route to the folder directory
        os.chdir(path)
        print(path)
        
        while(county < 21):
            # Capture video frame
            _, image_frame = vid_cam.read()

            # Convert frame to grayscale
            gray = image_frame

            # Detect frames of different sizes, list of faces rectangles
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            # Loops for each faces
            for (x, y, w, h) in faces:

                # Crop the image frame into rectangle
                cv2.rectangle(image_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Increment sample face image
                county += 1

                cv2.imwrite(str(county) + ".jpg", gray[y:y+h, x:x+w])
                # Display the video frame, with bounded rectangle on the person's face
                cv2.imshow('frame', image_frame)

            # To stop taking video, press 'q' for at least 100ms
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

            # If image taken reach 50, stop taking video
            elif county > 20:
                county = 0
                break

        # Stop video
        vid_cam.release()

        # Close all started windows
        cv2.destroyAllWindows()

        # Clear Matric Number
        matric_num == ""

        # Display on LCD "Successfully Enrolled Student_Name"
        lcd.text("Done!", 1)
        lcd.text("Enroll Successful!", 2)
        time.sleep(3)
        lcd.clear()
        Trainface()
        lcd.clear()
        lcd.text("Model Trained", 1)
    return

def ScanOldFaces():
    global count
    vid_cam = cv2.VideoCapture(0)
    while (count<=3):
        lcd.text("Scanning...", 1)
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
        
    # read input from the serial monitor
    newMatric=ser.readline()
    time.sleep(0.1)
    newMat=newMatric.decode('utf-8').rstrip()
    print("New Matric:",newMat)
    captureNewface(newMat)
    newMat=""