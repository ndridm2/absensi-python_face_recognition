import time
import cv2
import cv2.data
from database import fetchUser, storeAttendance

font = cv2.FONT_HERSHEY_SIMPLEX
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

recognition = True
ask = False
username = ''
userId = 0
notMe = []
saved = True
saving = False
start = 0
current_time = time.strftime('%d-%m-%Y %H:%M:%S')

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
                
                cv2.putText(frame, str(username), (x+5, y+5), font, 1, (255, 255, 255))

                if userId not in notMe:
                    recognition = False
                    ask = True
            else:
                cv2.putText(frame, str('unknown'), (x+5, y+5), font, 1, (255, 255, 255))

    if(saving):
        cv2.putText(frame, username + ' Attendaces', (30, 420), font, 0.8, (0, 255, 0))
        cv2.putText(frame, current_time, (30, 450), font, 0.6, (255, 255, 255))

        if(saved):
            storeAttendance(userId)
            cv2.imwrite('images/' + username + "-" + str(int(time.time())) + '.jpg', frame)
            saved = False
        
        # reset timer 2 second
        if(int(time.time()) > start):
            notMe = []
            recognition = True
            ask = False
            saving = False

    elif(ask):
        cv2.putText(frame, "Apakah kamu", (30, 350), font, 1, (255, 255, 255))
        cv2.putText(frame, username, (30, 385), font, 1, (255, 255, 255))
        cv2.putText(frame, "no (x)", (30, 430), font, 1, (255, 255, 255))
        cv2.putText(frame, "| yes (enter)", (145, 430), font, 1, (255, 255, 255))

    cv2.imshow('Face Recognition', frame)

    k = cv2.waitKey(10)
    if(k == 27): # esc
        break
    elif(k == 120): # x
        notMe.append(userId)
        recognition = True
        ask = False
    elif(k == 13): # enter
        saved = True
        saving = True
        start = int(time.time()) + 2 # 1:05:30 >  1:05:32

cam.release()
print("[INFO] Exit")
cv2.destroyAllWindows()

