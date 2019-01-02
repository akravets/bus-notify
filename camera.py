import cv2 as cv
import threading
import os
import os.path
import logging
from threading import Timer
import tensorflow as tf
from tensorflow import keras
import numpy as np

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
from keras.applications import vgg16

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
image_batch = []

def init():
   filename = 'bus.jpg'
   # load an image in PIL format
   original = load_img(filename, target_size=(32, 32))
   print('PIL image size',original.size)
   plt.imshow(original)
   plt.show()
   
   # convert the PIL image to a numpy array
   # IN PIL - image is in (width, height, channel)
   # In Numpy - image is in (height, width, channel)
   numpy_image = img_to_array(original)
   plt.imshow(np.uint8(numpy_image))
   plt.show()
   print('numpy array size',numpy_image.shape)
   
   # Convert the image / images into batch format
   # expand_dims will add an extra dimension to the data at a particular axis
   # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
   # Thus we add the extra dimension to the axis 0.
   global image_batch
   image_batch = np.expand_dims(numpy_image, axis=0)
   print('image batch size', image_batch.shape)
   plt.imshow(np.uint8(image_batch[0]))

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
   
   model.fit(train_images, train_labels, epochs=10)
   predictions = model.predict(test_images)
  
   processed_image = vgg16.preprocess_input(image_batch.copy())
 
   # get the predicted probabilities for each class
   predictions = model.predict(processed_image)
   print(predictions)
 
   # convert the probabilities to class labels
   # We will get top 5 predictions which is the default
   #label = decode_predictions(predictions)
   #print(label)

def read_from_stream():
   Timer(1.0,read_from_stream,[]).start()
   #vidcap = cv.VideoCapture('rtsp://root:azr26p@192.168.1.25:554/stream1')
   #success,image = vidcap.read()
   logger.info("got image")
   #cv.imwrite("frame.jpg", image)     # save frame as JPEG file     
   #analyze()

#read_from_stream()
init()
analyze()