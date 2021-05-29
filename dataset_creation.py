
import numpy as np
import os
import shutil
from tqdm.notebook import tqdm
import torch
import random
from utils import *
import datetime
import time
from utils import set_logger

logger,logger_t = set_logger()

t_set = lambda: datetime.datetime.now().astimezone()
t_diff = lambda t: str(t_set() - t)
t_stamp = lambda t=None: str(t) if t else str(t_set())


def createDataset(df,dataPath,destPath,ids,clip_len=50,size=60):
  random.seed(43)
  logger.info(f'Seed value is 43')
  assert size % 60 == 0, "Size should be in the multiple of 60"
  assert type(ids) == list, "expecting a list of strings"
  data_files = {i:[] for i in ids}
  for root,dir,files in os.walk(dataPath):
    for f in files:
      id = f.split('_')[0]
      if id in ids:
        data_files[id].append(os.path.join(root,f))
  for subset in range(size//60):
    t = t_set()
    for idx in range(len(list(data_files.keys()))):
      path = sorted(data_files[list(data_files.keys())[idx]])
      topic = (list(data_files.keys())[idx])
      while True:
        root_path = os.path.join(destPath, topic.split('-')[0], topic+'_'+str(subset))
        safedirs(root_path)
        rand_n = random.randint(0,len(path)-clip_len)
        frame_path = path[rand_n : rand_n+clip_len]
        labels = []
        for fp in frame_path:
          shutil.copy(fp, os.path.join(root_path, os.path.basename(fp)))
          #frame is the datapath with label at the end of the filename
          labels.append(int(fp.split('.')[0][-1]))
        if topic.split('-')[0] == 'adl':
          break
        elif topic.split('-')[0] == 'fall' and labels.count(1) > 4:
          break
        shutil.rmtree(root_path)
    logger.info(f'time elapsed for each subset {subset} is {t_diff(t)}')

