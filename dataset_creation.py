import numpy as np
import cv2
import os
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn as fasterrcnn
from tqdm.notebook import tqdm
import torch
from torch import cuda
from torch.utils.data import random_split,Dataset, DataLoader
import random


class clipDataset(Dataset):
  def __init__(self,df,dataPath,ids,clip_len=50):
    assert type(ids) == list, "expectint a list of strings"
    self.df = df
    self.clip_len = clip_len
    self.data_files = {i:[] for i in ids}
    for root,dir,files in os.walk(dataPath):
      for f in files:
        id = f.split('_')[0]
        if id in ids:
          self.data_files[id].append(os.path.join(root,f))
    random.seed(43)
  
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, index):
    files = self.data_files
    id = index % len(files)
    path = sorted(files[list(files.keys())[id]])
    rand_n = random.randint(0,len(path)-self.clip_len)
    frame_path = path[rand_n : rand_n+self.clip_len]
    labels = []
    frames = []
    for path in frame_path:
      #frame is the datapath with label at the end of the filename
      labels.append(int(path.split('.')[0][-1]))
      #read the image, convert to RGB and then to Tensor
      tensor_img = transforms.ToTensor()(cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB))
      frames.append(tensor_img)
    frames = torch.stack(frames,dim=0)
    return frames, torch.IntTensor(labels)