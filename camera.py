import cv2 as cv
import threading
import os
import os.path
import logging
from threading import Timer
import tensorflow as tf
from tensorflow import keras
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze():
   print(tf.__version__)
   dataset = keras.datasets.cifar10
   (train_images, train_labels), (test_images, test_labels) = dataset.load_data()

   train_images = train_images / 255.0

   test_images = test_images / 255.0

   model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32, 3)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)])

   model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
   
   model.fit(train_images, train_labels, epochs=5)
   predictions = model.predict(test_images)
   print(predictions)

def read_from_stream():
   Timer(1.0,read_from_stream,[]).start()
   #vidcap = cv.VideoCapture('rtsp://root:azr26p@192.168.1.25:554/stream1')
   #success,image = vidcap.read()
   logger.info("got image")
   #cv.imwrite("frame.jpg", image)     # save frame as JPEG file     
   #analyze()

#read_from_stream()
analyze()