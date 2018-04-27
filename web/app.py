#coding: utf-8

from uuid import uuid4

from flask import Flask, render_template, request, send_from_directory
from flask_uploads import UploadSet, configure_uploads, IMAGES

from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from scipy.misc import imresize
import numpy as np
import cv2

from gevent.wsgi import WSGIServer

app = Flask(__name__)

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'
configure_uploads(app, photos)

haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')

img_height, img_width, channels = 350, 350, 3
model = load_model("model/model.h5")

def detect_face(f_cascade, filepath, scaleFactor=1.1):
    img = load_img(filepath)
    img = imresize(img, size=(img_height, img_width))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)
    if len(faces) == 0:
        return np.zeros((1,1))
    x, y, w, h = faces[0]
    cropped_image = img[y:y + h, x:x + w, :]
    resized_image = cv2.resize(cropped_image, (img_height, img_width))
    filename = filepath.split('/')[-1]
    cv2.imwrite("uploads/cropped_{}".format(filename), resized_image)
    return resized_image

def get_score(filepath):
    test_x = detect_face(haar_face_cascade, filepath)
    if not test_x.any():
        return None
    test_x = test_x / 255.
    test_x = test_x.reshape((1,) + test_x.shape)
    predicted = model.predict(test_x)
    return round(predicted[0][0], 2)

@app.route("/")
def index():
    return render_template("upload.html", image_name="demo.jpeg", score=4.04)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'], name="{}.".format(str(uuid4())))
        score = get_score("uploads/{}".format(filename))
        if not score:
            return render_template("error.html")
        else:
            return render_template("upload.html", image_name="{}".format(filename), score=score)

@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory("uploads", filename)


if __name__ == '__main__':
    WSGIServer(('0.0.0.0', 5000), app).serve_forever()
