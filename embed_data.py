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


t_set = lambda: datetime.datetime.now().astimezone()
t_diff = lambda t: str(t_set() - t)
t_stamp = lambda t=None: str(t) if t else str(t_set())

def embedded_data(datapath):
  data_dict = defaultdict(list)
  for root,dir,files in os.walk(datapath):
      for f in files:
        frame_path = os.path.join(root,f)
        #print(root)
        folder = os.path.basename(root)
        data_dict[folder].append(frame_path)
  for key in list(data_dict.keys()):
    frames, labels = [], []
    for f in sorted(data_dict[key]):
      labels.append(int(f.split('.')[0][-1]))
      temp = f.split('/')
      mode = temp[3]
      filename = temp[5]
      img = cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2RGB)
      frames.append(img)
    frames = np.stack(frames,axis=0)
    t=t_set()
    image_embedding(frames, np.max(labels), mode, filename)
    logger.info(f"Total time taken for {key} is {t_diff(t)}")