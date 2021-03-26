import urllib.request as urlreq
import os
import cv2
import numpy as np
from pathlib import Path
from zipfile import ZipFile
import time
import random

import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from tqdm.notebook import tqdm
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn as fasterrcnn
from sklearn.preprocessing import minmax_scale


class PrepareData:
  def __init__(self,name='urfd',download=True,create_vectors=True):
    # create necessary folders to accomodate dataset
    wrk_dir = os.getcwd()
    #Parent directory
    if not os.path.exists(os.path.join(wrk_dir, 'dataI')):
      os.mkdir(os.path.join(wrk_dir, 'dataI'))
    self.data_dir = os.path.join(wrk_dir, 'dataI')
    #Directory containing Fall sequences
    if not os.path.exists(os.path.join(wrk_dir, 'dataI/fall_seq')):
      os.mkdir(os.path.join(wrk_dir, 'dataI/fall_seq'))
    self.fall_dir = os.path.join(wrk_dir, 'dataI/fall_seq')
    #Directory containing ADL sequences
    if not os.path.exists(os.path.join(wrk_dir, 'dataI/adl_seq')):
      os.mkdir(os.path.join(wrk_dir, 'dataI/adl_seq'))
    self.adl_dir = os.path.join(wrk_dir, 'dataI/adl_seq')
    self.labels = {}
    #download dataset
    if download:
      self.download(name=name)
    #whether to create vector files
    if create_vectors:
      self.create_vectors([self.fall_dir, self.adl_dir])
  
  def extract_zip(self, file_name, dir):
    '''
    Extracts the contents downloaded from the URL
    '''
    zip_ref = ZipFile(file_name)
    if dir  == 'fall':
      zip_ref.extractall(self.fall_dir)
    elif dir == 'adl':
      zip_ref.extractall(self.adl_dir)
    zip_ref.close()
    os.remove(file_name)

  
  def download(self,name='urfd'):
    '''
    Download the dataset from the URL
    '''
    if name == 'urfd':
      for n in tqdm(range(1,41)):
          # Add zeros to adapt files in Website
          if n < 10:
              n = '0'+str(n)
          #only 31 Fall sequences available
          if int(n)<31:
            #Load Fall sequences from the corresponding URL
            urlreq.urlretrieve(f'http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-{n}-cam0-rgb.zip',os.path.join(self.data_dir,f'fall-{n}-cam0-rgb.zip'))
            file_name = os.path.join(self.data_dir,f'fall-{n}-cam0-rgb.zip')
            #Extract elements into the respective folder
            self.extract_zip(file_name,'fall')
          #Load ADL sequences from the corresponding URL
          urlreq.urlretrieve(f'http://fenix.univ.rzeszow.pl/~mkepski/ds/data/adl-{n}-cam0-rgb.zip',os.path.join(self.data_dir,f'adl-{n}-cam0-rgb.zip'))
          file_name = os.path.join(self.data_dir,f'adl-{n}-cam0-rgb.zip')
          #Extract elements into respective folder
          self.extract_zip(file_name,'adl')
    else:
      print("the available datasets are ['urfd]")
  
  def create_vectors(self,folder):
    '''
    inputs:
    folder: convert images in the corresponding folder to vectors
    returns:
    dictionary containing filenames and labels
    Create vectors based on bounding box center
    '''
    assert type(folder) == list, "Expecting a list of directory path"
    print(folder)

    #location for Fall sequence bbox center vector
    fall_bbox = os.path.join(self.data_dir,'fall_bbox')
    if not os.path.exists(fall_bbox):
      os.mkdir(fall_bbox)

    #location for ADL sequence bbox center vector
    adl_bbox = os.path.join(self.data_dir,'adl_bbox')
    if not os.path.exists(adl_bbox):
      os.mkdir(adl_bbox)
    #load model for bbox generation
    model = fasterrcnn(pretrained=True)
    #utilize gpu if available
    if cuda.is_available():
      model = model.cuda()
    #inference mode
    model.eval()
    for dir in folder:
      if dir == self.fall_dir:
        name = 'fall_'
        label = 1
        bbox_file = fall_bbox
      else:
        dir = self.adl_dir
        label = 0
        name = 'adl_'
        bbox_file = adl_bbox

      l = os.listdir(dir)
      #iterate over the files in the directory
      print(f"creating {name[:-1]} files")
      for i in tqdm(range(len(l))):
        temp = []
        start = time.time()
        for j in sorted(os.listdir(os.path.join(dir,l[i]))):
          img_path = os.path.join(dir,l[i],j)
          image = cv2.imread(img_path)
          image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
          #convert image to tensor for model
          tfm_img = transforms.ToTensor()(image)
          #move to gpu
          if cuda.is_available():
            tfm_img = tfm_img.cuda()
          #make inference and returns dict containing "boxes,labels,scores"
          predictions = model([tfm_img])
          #predicted labels
          labels = predictions[0]['labels'].cpu().detach().numpy()
          #model predicts '1' label for person
          person_box_coord = np.where(labels==1)[0]
          #if many boxes are available, then we choose one with high score
          if person_box_coord.size > 1:
            id = np.argmax([predictions[0]['scores'][i].cpu().detach().numpy() for i in person_box_coord])
            id = person_box_coord[id]
          #if there is no person in the frame
          elif person_box_coord.size < 1:
            continue
          else:
            id = person_box_coord[0]
          #get the bbox coordinates
          box_coord = predictions[0]['boxes'][id].unsqueeze(0).cpu().detach().numpy().astype('int')
          #calculate bbox center point
          for x1,y1,x2,y2 in box_coord:
            box_center = (int(x1+(x2-x1)/2),int(y1+(y2-y1)/2))
            temp.append(box_center)
        print('elapsed time: ',time.time() - start,end='\r')
        #save the numpy array/image in a file for future processing 
        file_path = os.path.join(bbox_file, name+f'{i}.npy')
        np.save(file_path,np.array(temp))
        self.labels[file_path] = label

class URFDDataset(Dataset):
  '''
  Create PyTorch dataset class 
  '''
  def __init__(self,labels,clip_len=50):
    #Labels dict of filenames and label
    self.labels = labels
    self.fnames = list(labels.keys())
    #Frame width
    self.clip_len = clip_len
    #shuffle before splitting
    random.shuffle(self.fnames)

  def normalize(self,buffer):
    #Normalize to range of 0-1
    new_buffer = minmax_scale(buffer)
    return new_buffer

  def __getitem__(self,index):
    fname = self.fnames[index]
    label = self.labels[fname]
    vectors = np.load(fname)
    #Normalize the values to 0-1
    vectors = self.normalize(vectors)
    #Random window 
    n = np.random.randint(0,len(vectors) - self.clip_len)
    buffer = vectors[n:n + self.clip_len]
    #Convert to tensor for processing
    buffer = transforms.ToTensor()(buffer)
    return buffer,label
    
  def __len__(self):
    #return len of the data
    return len(self.fnames)