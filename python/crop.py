import numpy as np
from sklearn.datasets import fetch_openml
import joblib
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import csv
import cv2
from skimage import exposure
from skimage import transform

try:
    os.mkdir(os.getcwd()+'\Dataset\Exposed')
except:
    pass

br = 0

with open(os.getcwd()+'\Dataset\Train.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if(br != 0):
            try:
                os.mkdir(os.getcwd()+'\Dataset\Exposed\\'+row[7].split("/")[1])
            except:
                pass
            img = cv2.imread(os.getcwd()+'\Dataset\Train\\'+row[7].split("/")[1]+'\\'+row[7].split("/")[2])
            image = transform.resize(img, (30, 30))
            image = np.array(image, dtype=np.float32)
            exposed_img = exposure.equalize_adapthist(image, clip_limit=0.01)
            cv2.imwrite(os.getcwd()+'\Dataset\Exposed\\'+row[7].split("/")[1]+'\\'+row[7].split("/")[2], exposed_img)
        print(br)
        br += 1

print("Loaded train data")