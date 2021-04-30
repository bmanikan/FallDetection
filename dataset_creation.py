
import numpy as np
import os
import shutil
from tqdm.notebook import tqdm
import cv2
from torchvision import transforms
from tqdm.notebook import tqdm
import torch
from torch import cuda
from torch.utils.data import random_split,Dataset, DataLoader
import random
from collections import defaultdict


def createDataset(df,dataPath,destPath,ids,clip_len=50):
  random.seed(43)
  assert type(ids) == list, "expecting a list of strings"
  df = df
  destPath = destPath
  clip_len = clip_len
  data_files = {i:[] for i in ids}
  for root,dir,files in os.walk(dataPath):
    for f in files:
      id = f.split('_')[0]
      if id in ids:
        data_files[id].append(os.path.join(root,f))
  for idx in range(len(list(data_files.keys()))):
    path = sorted(data_files[list(data_files.keys())[idx]])
    topic = (list(data_files.keys())[idx])
    while True:
      safedirs(os.path.join(destPath, topic.split('-')[0], topic))
      rand_n = random.randint(0,len(path)-clip_len)
      frame_path = path[rand_n : rand_n+clip_len]
      labels = []
      for fp in frame_path:
        shutil.copy(fp, os.path.join(destPath, topic.split('-')[0], topic, os.path.basename(fp)))
        #frame is the datapath with label at the end of the filename
        labels.append(int(fp.split('.')[0][-1]))
      if topic.split('-')[0] == 'adl':
        break
      elif topic.split('-')[0] == 'fall' and labels.count(1) > 4:
        break
      shutil.rmtree(os.path.join(destPath,topic.split('-')[0],topic))




class clipDataset(Dataset):
  def __init__(self,datapath,mode='train'):
    self.datapath = datapath
    self.data_dict = defaultdict(list)
    for root,dir,files in os.walk(datapath):
      for f in files:
        frame_path = os.path.join(root,f)
        folder = os.path.basename(root)
        if mode in frame_path:
          self.data_dict[folder].append(frame_path)
  
  def __len__(self):
    return len(self.data_dict)
  
  def __getitem__(self,index):
    key = list(self.data_dict.keys())[index]
    frames = []
    labels = []
    files_list = self.data_dict[key]
    for f in files_list:
      labels.append(int(f.split('.')[0][-1]))
      print(labels)
      #read the image, convert to RGB and then to Tensor
      tensor_img = transforms.ToTensor()(cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2RGB))
      frames.append(tensor_img)
    frames = torch.stack(frames,dim=0)
    frames,labels = image_embedding(frames.unsqueeze(0), torch.IntTensor(labels).unsqueeze(0))
    return frames.squeeze(0),labels.squeeze(0)

