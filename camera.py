import cv2 as cv
import threading
import os
import os.path
import logging
from threading import Timer
#import subprocess

COUNT = 0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#subprocess.call("python", "classify_image.py","--image_file", "bus.jpg")

def read_from_stream():
   global COUNT
   Timer(1.0,read_from_stream,[]).start()
   vidcap = cv.VideoCapture('rtsp://root:azr26p@192.168.1.25:554/stream1')
   success,image = vidcap.read()
   logger.info("got image")
   cv.imwrite("frame.jpg", image)     # save frame as JPEG file     
   os.system("/camera/models-master/tutorials/image/imagenet/classify_image.py --image_file=frame.jpg")
   COUNT += 1

read_from_stream()