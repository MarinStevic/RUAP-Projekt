from flask import Flask, request, render_template, jsonify
from PIL import Image, ImageOps
import numpy as np
import os
import csv
import cv2
from tensorflow.keras.models import model_from_json

app = Flask(__name__, template_folder=os.getcwd()+'\\templates')

# load json and create model
json_file = open(os.getcwd()+'\model\model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(os.getcwd()+'\model\model.h5')
print('Loaded model from disk')

# load labels for each class
labels = []

with open(os.getcwd()+'\Dataset\Labels.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        for label in row:
            labels.append(label)

@app.route('/json/', methods=['GET','POST'])
def json():
    if request.method == 'POST':
        # check if the post request has the file part
        if ('file' not in request.files):
           return render_template('json.html', error='someting went wrong...')

        user_file = request.files['file']
        temp = request.files['file']
        if (user_file.filename == ''):
           return render_template('json.html', error='file not found...')
        else:
            path = os.path.join(os.getcwd()+'\\static\\'+user_file.filename)
            user_file.save(path)

        data = []
        height = 30
        width = 30

        image = cv2.imread(path)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((height, width))
        data.append(np.array(size_image))

        X_test = np.array(data)
        X_test = X_test.astype('float32')/255 
        pred = model.predict_classes(X_test)
        return jsonify({"prediction":{"classID":int(pred[0]),"label":str(labels[pred[0]])}})
    return render_template('json.html')

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        # check if the post request has the file part
        if ('file' not in request.files):
           return render_template('index.html', error='someting went wrong...')

        user_file = request.files['file']
        temp = request.files['file']
        if (user_file.filename == ''):
           return render_template('index.html', error='file not found...')
        else:
            path = os.path.join(os.getcwd()+'\\static\\'+user_file.filename)
            user_file.save(path)

        data=[]
        height = 30
        width = 30

        image = cv2.imread(path)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((height, width))
        data.append(np.array(size_image))

        X_test = np.array(data)
        X_test = X_test.astype('float32')/255 
        pred = model.predict_classes(X_test)
        return render_template('index.html', pClass=str(pred[0]), pLabel=str(labels[pred[0]]))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=3000, debug=False, host='0.0.0.0')