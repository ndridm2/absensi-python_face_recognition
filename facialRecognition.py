import cv2
import cv2.data
import os
import numpy as np
from database import fetchUser, storeAttendance

font = cv2.FONT_HERSHEY_COMPLEX
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

recognition = True
ask = False
username = ''
userId = 0

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

faceDetector = cv2.CascadeClassifier(cascade_path)

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

while True:
    ret, frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(gray)

    if(recognition):
        for(x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0))

            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if(confidence < 100):
                user = fetchUser(id)
                userId = user['id']
                username = user['name']
                
                cv2.putText(frame, str(username) + str(userId), (x+5, y+5), font, 1, (255, 255, 255))

                recognition = False
                ask = True
            else:
                cv2.putText(frame, "Apakah kamu", (30, 350), font, 1, (255, 255, 255))
                cv2.putText(frame, username, (30, 385), font, 1, (255, 255, 255))
                cv2.putText(frame, "tidak (x)", (30, 430), font, 1, (255, 255, 255))
                cv2.putText(frame, "ya (enter)", (180, 430), font, 1, (255, 255, 255))

    if(ask):
        cv2.putText(frame, str('unknown'), (x+5, y+5), font, 1, (255, 255, 255))

    cv2.imshow('Face Recognition', frame)

    k = cv2.waitKey(10)
    if(k == 27):
        break
    elif(k == 120): #"X"
        recognition = True
    elif(k == 13):
        storeAttendance(userId)

print("[INFO] Exit")
cam.release()
cv2.destroyAllWindows()

