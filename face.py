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
        
        if(result[0] == 1):
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(frame,'rebecca',(10,200), font, 4,(255,0,0),2,cv.LINE_AA)
        else:
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(frame,'wisnu',(10,300), font, 4,(255,0,0),2,cv.LINE_AA)

    cv.imshow('coba', frame)

    ch = cv.waitKey(1)

    if ch & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

