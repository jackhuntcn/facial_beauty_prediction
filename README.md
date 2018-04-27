# facial_beauty_prediction

A Facial Beauty Prediction web demo, using dataset from SCUT-FBP5500-Database.
Inspire by https://towardsdatascience.com/how-attractive-are-you-in-the-eyes-of-deep-neural-network-3d71c0755ccc
MSE loss on test data is 0.16.

## How to run this demo

```
git clone https://github.com/jackhuntcn/facial_beauty_prediction.git
cd facial_beauty_prediction/web/
# download model.h5 from BaiDuYun and put it into model/ directory, make sure model/model.h5 is exists
# https://pan.baidu.com/s/1oJd67FMwO9KVwN5hgda3JQ password: mw8m
python app.py
```

## requirements

* numpy
* scipy
* cv2
* keras 2.1.5
* flask / flask-uploads
* gevent
