from PIL import Image, ImageOps
import numpy as np
import os
import csv
import cv2
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import pandas as pd

height = 30
width = 30

# load json and create model
json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights('model/model.h5')
print('Loaded model from disk')

# Predicting with the test data
y_test = pd.read_csv(os.getcwd()+'\Dataset\Test.csv')
labels = y_test['Path'].to_numpy()
y_test = y_test['ClassId'].values

data = []

for f in labels:
    image=cv2.imread(os.getcwd()+'\Dataset\Test\\'+f.replace('Test/', ''))
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((height, width))
    data.append(np.array(size_image))

X_test = np.array(data)
X_test = X_test.astype('float32')/255 
pred = model.predict_classes(X_test)

#Accuracy with the test data
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred))