import os
import cv2
import cv2.data
import numpy as np
from PIL import Image

path = 'dataset'
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

recognizer = cv2.face.LBPHFaceRecognizer_create()

detector = cv2.CascadeClassifier(cascade_path)

def getImageAndLabels(path):

    faceSamples = []
    ids = []

    dirs = os.listdir(path)

    for dir in dirs:

        fullPath = os.path.join(path, dir)
        images = [os.path.join(fullPath, f) for f in os.listdir(fullPath)]

        for image in images:
            PILImg = Image.open(image)

            imgNumpy = np.array(PILImg, 'uint8')

            faces = detector.detectMultiScale(imgNumpy)
            
            for (x, y, w, h) in faces:
                faceSamples.append(imgNumpy[y:y+h, x:x+w])
                ids.append(int(dir))
    
    return faceSamples, ids

def training():
    faces, ids =  getImageAndLabels(path)
    recognizer.train(faces, np.array(ids))
    recognizer.save('trainer/trainer.yml')

# faces, ids =  getImageAndLabels(path)
# recognizer.train(faces, np.array(ids))
# recognizer.save('trainer/trainer.yml')

# print("\n [INFO] {0} faces trained. Exiting program".format(len(np.unique(ids))) )