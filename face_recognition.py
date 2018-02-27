import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join

# dapatkan data train wisnu
data_path_wisnu = './data_train/wisnudj/'
only_files_wisnu =  [f for f in listdir(data_path_wisnu) if isfile(join(data_path_wisnu, f))]

training_datas_wisnu, labels_wisnu = [], []

for i, files in enumerate(only_files_wisnu):
    image_path = data_path_wisnu + only_files_wisnu[i]
    images = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    training_datas_wisnu.append(np.asarray(images, dtype=np.uint8))
    labels_wisnu.append('wisnu')

labels_wisnu = np.asarray(labels_wisnu, dtype=np.int32)

model_wisnu = cv.face.LBPHFaceRecognizer_create()

model_wisnu.train(np.asarray(training_datas_wisnu), np.asarray(labels_wisnu))

# dapatkan data train rebecca
data_path_rebecca = './data_train/rebecca/'
only_files_rebecca =  [f for f in listdir(data_path_rebecca) if isfile(join(data_path_rebecca, f))]

training_datas_rebecca, labels_rebecca = [], []

for i, files in enumerate(only_files_rebecca):
    image_path = data_path_rebecca + only_files_rebecca[i]
    images = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    training_datas_rebecca.append(np.asarray(images, dtype=np.uint8))
    labels_rebecca.append(i)

labels_rebecca = np.asarray(labels_rebecca, dtype=np.int32)

model_rebecca = cv.face.LBPHFaceRecognizer_create()

model_rebecca.train(np.asarray(training_datas_rebecca), np.asarray(labels_rebecca))
print('model berhasil di training')


# RUN facial recognition

face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img):

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return img, []

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y+h, x:x+w]
        roi = cv.resize(roi, (200, 200))

    return img, roi

cap = cv.VideoCapture(0)

while True:

    ret, frame = cap.read()

    image, face = face_detector(frame)

    if face != []:

        face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
        results_wisnu = model_wisnu.predict(face)
        results_rebecca = model_rebecca.predict(face)

        wisnu = int(100 * (1 - (results_wisnu[1]) / 300))
        rebecca = int(100 * (1 - (results_rebecca[1]) / 300))

        print(results_wisnu)
        

        if wisnu > rebecca:
            name = 'Wisnu'
        else:
            name = 'rebecca'

        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(image,name,(10,500), font, 4,(255,0,0),2,cv.LINE_AA)

        print(wisnu, rebecca)


    cv.imshow('coba', image)

    ch = cv.waitKey(1)

    if ch & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()