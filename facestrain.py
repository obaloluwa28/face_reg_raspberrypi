import cv2
import os
import numpy as np
from PIL import Image
import pickle

# def Trainface
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier(
    '/home/pi/Desktop/face rec/New folder/OpenCV-Face-Recognition-Python/src/cascades/data/haarcascade_frontalface_alt.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 1
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)

            label = os.path.basename(root).replace(" ", "-").lower()
            print(label, path)
            # if the folder name(Label) doesnet already exist in the Label_id dictionary
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            print("Id:", id_)

            pil_image = Image.open(path).convert("L")  # grayscale
            image_array = np.array(pil_image, "uint8")
            faces = face_cascade.detectMultiScale(
                image_array, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

with open("/home/pi/Desktop/face rec/New folder/OpenCV-Face-Recognition-Python/src/pickles/face-labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("/home/pi/Desktop/face rec/New folder/OpenCV-Face-Recognition-Python/src/recognizers/face-trainner.yml")
