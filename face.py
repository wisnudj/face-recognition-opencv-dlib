import train_data
import face_detector
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

model = train_data.lbph('data_train')

while True:

    ret, frame = cap.read()

    img, faces = face_detector.haarcascade(frame)

    for face in faces:
        gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
        result = model.predict(gray)
        print(result[0], int(100 * (1 - (result[1]) / 300)))

    cv.imshow('coba', frame)

    ch = cv.waitKey(1)

    if ch & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

