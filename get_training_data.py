import numpy as np
import cv2 as cv
import imutils
import dlib
from imutils import face_utils
from os import listdir
from os.path import isfile, join

# fungsi mendeteksi kotak roi wajah dari library dlib
detector = dlib.get_frontal_face_detector()


def face_extractor(image):

  image = imutils.resize(image, width=500)
  gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

  #temukan wajah
  faces = detector(gray, 1)

  print(len(faces))

  if len(faces) == 0:
    return None

  # mendapatkan koordinat untuk mendapat kotak wajah dengan konversi faces
  (x, y, w, h) = face_utils.rect_to_bb(faces[0])

  cropped_face = image[y:y+h, x:x+w]

  return cropped_face

  
cap = cv.VideoCapture(0)
count = 0

# dapatkan 30 foto dari camera
while count <= 30:

  ret, frame = cap.read()

  if face_extractor(frame) is not None:
    face = face_extractor(frame)
    count = count + 1

    # save to direktori dengan nama unik
    file_name_path = './data_train/' + str(count) + '.jpg'
    cv.imwrite(file_name_path, face)
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(frame,str(count),(10,500), font, 4,(255,0,0),2,cv.LINE_AA)
  else:
    print('face not found')
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(frame,'face not found',(10,500), font, 4,(255,0,0),2,cv.LINE_AA)
    pass

  cv.imshow('potret', frame)

  ch = cv.waitKey(1)

  if ch & 0xFF == ord('q'):
    break

cap.release()
cv.destroyAllWindows()
print('kelar')

