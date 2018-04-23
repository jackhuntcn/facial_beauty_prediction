#coding: utf-8

from uuid import uuid4

from flask import Flask, render_template, request, send_from_directory
from flask_uploads import UploadSet, configure_uploads, IMAGES

from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from scipy.misc import imresize

app = Flask(__name__)

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'
configure_uploads(app, photos)

img_height, img_width, channels = 350, 350, 3
model = load_model("model/model.h5")

def get_score(filename):
    img = load_img(filename)
    img = imresize(img, size=(img_height, img_width))
    test_x = img_to_array(img).reshape(img_height, img_width, channels)
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
        return render_template("upload.html", image_name=filename, score=score)

@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory("uploads", filename)


if __name__ == '__main__':
    app.run(host="0.0.0.0")
