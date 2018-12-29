import cv2 as cv
import threading
import os
import os.path
from threading import Timer

def vars():
   global file_name
   file_name = "frame.jpg"

def read_from_stream():
   Timer(5.0,read_from_stream,[]).start()
   if(os.path.exists(file_name)):
       os.remove(file_name)
   vidcap = cv.VideoCapture('rtsp://root:azr26p@192.168.1.25:554/stream1')
   success,image = vidcap.read()
   cv.imwrite(file_name, image)     # save frame as JPEG file     
   print("test")

vars()
read_from_stream()