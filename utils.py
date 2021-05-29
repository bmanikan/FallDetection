from collections import defaultdict
from skimage.metrics import structural_similarity
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time,os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import datetime
import time
import logging
import sys


t_set = lambda: datetime.datetime.now().astimezone()
t_diff = lambda t: str(t_set() - t)
t_stamp = lambda t=None: str(t) if t else str(t_set())

root = os.getcwd()

def set_logger():
  logging.basicConfig(level=logging.INFO)
  logger_t = logging.getLogger('withoutlevel')
  logger = logging.getLogger('withlevel')

  f_handler = logging.FileHandler(root+'/Falldetection.log')
  f_handler_t = logging.FileHandler(root+'/Falldetection.log')

  f_format = logging.Formatter('%(levelname)s - %(message)s')
  f_format_t = logging.Formatter('%(message)s')

  f_handler.setFormatter(f_format)
  f_handler_t.setFormatter(f_format_t)

  logger.addHandler(f_handler)
  logger_t.addHandler(f_handler_t)
  return logger, logger_t


def safedirs(path):
  if not os.path.exists(path):
    os.makedirs(path)

def person_detector(imageMatrix,label,mode,folder):
  '''
  Moving object detection by frame differencing
  input: torch tensor object of shape [batch,50,3,480,640] and labels array
  output: tuple of torch tensor [batch,1,224,224] and label object 
  '''
  info = ['adl','fall']
  bbox_path = os.path.join(os.getcwd(), 'boundingBox', mode, info[label], folder)
  logger_t.info(f'Working on creating ... {bbox_path} set')
  safedirs(bbox_path)
  temp_img = imageMatrix
  center_points = []
  for i in range(0,temp_img.shape[0] - 2):
    img1 = cv2.medianBlur(cv2.cvtColor(temp_img[i],cv2.COLOR_BGR2GRAY),5)
    img2 = cv2.medianBlur(cv2.cvtColor(temp_img[i+2],cv2.COLOR_BGR2GRAY),5)
    (score,diff) = structural_similarity(img1,img2,full=True)
    diff = (diff * 255).astype('uint8')
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    dilate_frame = cv2.dilate(thresh, None, iterations=5)
    contours = cv2.findContours(dilate_frame.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1] 
    val = np.argmax([cv2.contourArea(c) for c in contours])
    x,y,w,h = cv2.boundingRect(contours[val])
    cv2.rectangle(img1, (x, y), (x + w, y + h), (36,255,12), 2)
    path = os.path.join(bbox_path, f'image_{i}.jpg')
    cv2.imwrite(path, img1)
    center = (int((x+w)/2), int((y+h)/2))
    center_points.append(center)
  return center_points

def get_transform():
  transform = tf.keras.Sequential([
                                   preprocessing.Resizing(224,224),
                                   preprocessing.Normalization()
  ])
  return transform


def image_embedding(images,label,mode,filename):
  transform = get_transform()
  info = ['adl', 'fall']
  trajectory_path = os.path.join(os.getcwd(), 'trajectories', mode, info[label])
  safedirs(trajectory_path)
  logger_t.info(f'creating trajectory image at {trajectory_path}')
  bkg = np.zeros((480,640),np.uint8)
  t=t_set()
  points = person_detector(images,label,mode,filename) 
  logger.info(f'Elapsed time for Bounding box calculation is {t_diff(t)}')
  for p in range(len(points)-1):
    bkg = cv2.line(bkg,points[p],points[p+1],(255,0,0),5)
  bkg = transform(tf.expand_dims(bkg,2))
  path = os.path.join(trajectory_path, filename+'.jpg')
  cv2.imwrite(path, bkg.numpy())
