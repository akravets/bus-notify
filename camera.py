import cv2 as cv
import threading
import os
import os.path
from threading import Timer

COUNT = 0

def read_from_stream():
   global COUNT
   Timer(1.0,read_from_stream,[]).start()
   vidcap = cv.VideoCapture('rtsp://root:azr26p@192.168.1.25:554/stream1')
   success,image = vidcap.read()
   cv.imwrite("frame%d.jpg" % COUNT, image)     # save frame as JPEG file     
   COUNT += 1
   print(COUNT)

read_from_stream()