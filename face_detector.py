import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join

def haarcascade(img):

    face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return img, []
    
    list_of_faces = []

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        list_of_faces.append(cv.resize(img[y:y+h, x:x+w], (200, 200)))
        
    return img, list_of_faces