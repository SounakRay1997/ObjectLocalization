from keras.models import load_model
import cv2
import numpy as np
import os
import pandas as pd
import keras
import tensorflow as tf
from keras import backend as K
from keras.applications.mobilenet import preprocess_input
from keras import Model
IMAGE_SIZE=224
image_height=480.0
image_width=640.0
start=0
def imageFilePaths(paths):
        text_files = [f for f in os.listdir(paths) if (f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp') or f.endswith('.jpeg') or f.endswith('.tif') or f.endswith('.gif') or f.endswith('.PNG') or f.endswith('.JPG') or f.endswith('.BMP') or f.endswith('.JPEG') or f.endswith('.TIF') or f.endswith('.GIF'))]
        for i in range(len(text_files)):
            text_files[i]=paths+text_files[i]
        return text_files
        
def log_mse(y_true, y_pred):
    return tf.reduce_mean(tf.log1p(tf.squared_difference(y_pred, y_true)), axis=-1)  
          
path='/home/sounak/FlipkartGRiDLevel3/test/'
imgList=imageFilePaths(path)
model=load_model('/home/sounak/FlipkartGRiDLevel3/model-0.82_1.h5', custom_objects={'log_mse':log_mse})
model1=load_model('/home/sounak/FlipkartGRiDLevel3/model-0.83.h5', custom_objects={'log_mse':log_mse})
model2=load_model('/home/sounak/FlipkartGRiDLevel3/model-0.81_2.h5', custom_objects={'log_mse':log_mse})
print(len(imgList))
f1=open("/home/sounak/FlipkartGRiDLevel3/submission_ensemble_3model.txt", "w+")
f1.write("image_name, x1, x2, y1, y2\r\n")
for f in imgList:
    print(f)
    start=start+1
    print(start)
    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = preprocess_input(np.array(img, dtype=np.float32))
    img = np.expand_dims(img, axis=0)
    region = model.predict(img)
    region = np.maximum(region, 0)
    region1 = model1.predict(img)
    region1 = np.maximum(region1, 0)
    region2 = model2.predict(img)
    region2 = np.maximum(region2, 0)
    print (region)
    print (region1)
    print (region2)
    x1_1 = np.int32(region[0][0] * image_width / IMAGE_SIZE)
    x2_1 = np.int32(region[0][1] * image_width / IMAGE_SIZE)
    y1_1 = np.int32(region[0][2] * image_height / IMAGE_SIZE)
    y2_1 = np.int32(region[0][3] * image_height / IMAGE_SIZE)
    x1_2 = np.int32(region1[0][0] * image_width / IMAGE_SIZE)
    x2_2 = np.int32(region1[0][1] * image_width / IMAGE_SIZE)
    y1_2 = np.int32(region1[0][2] * image_height / IMAGE_SIZE)
    y2_2 = np.int32(region1[0][3] * image_height / IMAGE_SIZE)
    x1_3 = np.int32(region2[0][0] * image_width / IMAGE_SIZE)
    x2_3 = np.int32(region2[0][1] * image_width / IMAGE_SIZE)
    y1_3 = np.int32(region2[0][2] * image_height / IMAGE_SIZE)
    y2_3 = np.int32(region2[0][3] * image_height / IMAGE_SIZE)
    c=f.rfind('/')
    name=f[c+1:]
    x1=(x1_1+x1_2+x1_3)/3
    x2=(x2_1+x2_2+x2_3)/3
    y1=(y1_1+y1_2+y1_3)/3
    y2=(y2_1+y2_2+y2_3)/3
    x1=np.int32(x1)
    x2=np.int32(x2)
    y1=np.int32(y1)
    y2=np.int32(y2)
    if (x2>640):
        x2=640
    if (y2>480):
        y2=480
    print (x1)
    print (x2)
    print (y1)
    print (y2)
    f1.write(name+",%d,%d,%d,%d\r\n" %(x1, x2, y1, y2))
f1.close
