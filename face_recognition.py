import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join

# dapatkan data train
data_path = './data_train/'
only_files =  [f for f in listdir(data_path) if isfile(join(data_path, f))]

training_datas, labels = [], []

images = cv.imread('wisnudj.jpg')

cv.imshow('sasa', images)

cv.waitKey(0)

cv.destroyAllWindows()