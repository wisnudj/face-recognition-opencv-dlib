import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join

# fungsi yang nge return model train dengan algoritma lbph
def lbph(dir_train_image):

    list_of_folder_image = listdir(dir_train_image)

    # dapatkan training data dan label
    labels, training_datas = [], []

    for i in list_of_folder_image:
        list_of_image = listdir(dir_train_image +'/' + i)
        
        for j, image in enumerate(list_of_image):
            
            image_path = dir_train_image + '/' + i + '/' + image
            
            image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

            training_datas.append(np.asarray(image, dtype=np.uint8))
            
            labels.append(i)

    labels = np.asarray(labels, dtype=np.int32)
    
    model = cv.face.LBPHFaceRecognizer_create()

    model.train(np.asarray(training_datas), np.asarray(labels))

    return model
