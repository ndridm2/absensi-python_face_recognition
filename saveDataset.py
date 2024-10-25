import cv2
import cv2.data
import os


# Pastikan haarcascade ada di direktori yang sama atau berikan path lengkap
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    print(f"File '{cascade_path}' tidak ditemukan. Pastikan file haarcascade ada di direktori yang sama.")
    exit()

cam = cv2.VideoCapture(0)
# Set video width dan height
cam.set(3, 640)  # lebar video
cam.set(4, 480)  # tinggi video

faceDetector = cv2.CascadeClassifier(cascade_path)

faceId = input('\n Enter user id and press <return> ==> ')

count = 0
while True:
    ret, frame = cam.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(gray)

    pathDataset = 'dataset/' + str(faceId) + '/'
    if not os.path.exists(pathDataset):
        os.makedirs(pathDataset)

    # proses penyimpanan gambar
    for (x, y, w, h) in faces:
        count += 1
        cv2.imwrite(pathDataset + str(count) + '.jpg', gray[y:y+h, x:x+w])

    # Gambar kotak di sekitar wajah yang terdeteksi
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('frame', frame)

    k = cv2.waitKey(100) & 0xff
    if k == 27:  # tekan ESC untuk keluar
        break
    elif count == 30:
        break

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
