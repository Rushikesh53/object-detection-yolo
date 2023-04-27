# print('Shri Ganesh')
from flask import Flask, render_template, request, session
import streamlit as st;
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# creating an upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# UPDATED_FOLDER = os.path.join('static', 'update')
# os.makedirs(UPDATED_FOLDER, exist_ok=True)


# defining allowed files
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# # configuring the upload folder for the applition
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['UPDATED_FOLDER'] = UPDATED_FOLDER
app.secret_key = os.urandom(12)

@app.route('/')
def index():
        return render_template('home.html')

@app.route('/', methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        # Upload file to flask
        uploaded_img = request.files['uploaded-file']
        # Extractacting uploaded data filename
        img_filename = secure_filename(uploaded_img.filename)
        # Uploading file to database 
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        # Storing uploaded file path in flask session
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)

        imgpath = img_filename
        img=cv2.imread(imgpath)

        classNames=[]
        classFile='coco.names.txt'

        with open(classFile,'rt') as f:
            classNames=f.read().rstrip('\n').split('\n')
        # print(classNames)

        configPath='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightPath='frozen_inference_graph.pb'

        net=cv2.dnn_DetectionModel(weightPath,configPath)
        net.setInputSize(320,320)
        net.setInputScale(1.0/127.5)
        net.setInputMean((127.5,127.5,127.5))
        net.setInputSwapRB(True)


        classIds,confs,bbox=net.detect(img,confThreshold=float(0.3))
        print(classIds,bbox)

        for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=1)
            cv2.putText(img,classNames[classId-1],(box[0],box[1]),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)

        # cv2.imshow('Output',img)
        outpath = "output.jpg"

        path = 'static/output/'
        cv2.imwrite(os.path.join(path , 'output.jpg'), img)


        # cv2.imwrite(outpath, img)
        cv2.waitKey(0)

        return render_template('home2.html')


@app.route('/show_image')
def displayImage():

    image_names = os.listdir('E:/new/object_detection_final/static/output')
    return render_template('show_image.html', image_name=image_names)

    # # retrieving uploaded file path from session
    # img_file_path = session.get('output', None)
    # #display image in flask application web page
    # return render_template('show_image.html', user_image = img_file_path)



if __name__=='__main__':
    app.run(debug = True)




# img=cv2.imread('road.jpeg')

# classNames=[]
# classFile='coco.names.txt'

# with open(classFile,'rt') as f:
#     classNames=f.read().rstrip('\n').split('\n')
# # print(classNames)

# configPath='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# weightPath='frozen_inference_graph.pb'

# net=cv2.dnn_DetectionModel(weightPath,configPath)
# net.setInputSize(320,320)
# net.setInputScale(1.0/127.5)
# net.setInputMean((127.5,127.5,127.5))
# net.setInputSwapRB(True)


# classIds,confs,bbox=net.detect(img,confThreshold=0.3)
# print(classIds,bbox)

# for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
#     cv2.rectangle(img,box,color=(0,255,0),thickness=1)
#     cv2.putText(img,classNames[classId-1],(box[0],box[1]),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)

# # displaying image
# cv2.imshow('Output',img)
# cv2.waitKey(0)




