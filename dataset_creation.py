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
  def __init__(self,df,dataPath,mode='train',clip_len=50):
    '''
    partition the dataset into train and test
    '''
    self.df = df
    self.mode = mode
    self.clip_len = clip_len
    ids = df[0].unique()
    train_size = int(0.8*len(ids))
    test_size = len(ids) - train_size
    train_ids,test_ids = random_split(ids,[train_size,test_size],generator=torch.Generator().manual_seed(43))
    self.train_files,self.test_files = {i:[] for i in train_ids},{i:[] for i in test_ids}
    for root,dir,files in os.walk(dataPath):
      for f in files:
        id = f.split('_')[0]
        if id in train_ids:
          self.train_files[id].append(os.path.join(root,f))
        elif id in test_ids:
          self.test_files[id].append(os.path.join(root,f))
    random.seed(43)
  
  def __len__(self):
    '''
    returns length
    '''
    return len(self.df)
  
  def __getitem__(self, index):
    if self.mode == 'train':
      files = self.train_files
    elif self.mode == 'test':
      files = self.test_files
    else:
      raise Exception("invalid mode")
    print(files.keys())
    id = index % len(files)
    path = files[list(files.keys())[id]]
    rand_n = random.randint(0,len(path)-self.clip_len)
    print(rand_n)
    frame_path = path[rand_n : rand_n+self.clip_len]
    print(frame_path)
    labels = []
    frames = []
    for path in frame_path:
      #frame is the datapath with label at the end of the filename
      labels.append(path.split('.')[0][-1])
      #read the image, convert to RGB and then to Tensor
      tensor_img = transforms.ToTensor()(cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB))
      frames.append(tensor_img)
    frames = torch.stack(frames,dim=0)
    return frames, labels