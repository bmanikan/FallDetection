import urllib.request as urlreq
import os
import cv2
import numpy as np
from pathlib import Path
import torch
from torch import cuda
from tqdm.notebook import tqdm
from zipfile import ZipFile
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn as fasterrcnn
import time

class Dataset:
  def __init__(self,name='urfd'):
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
    #download dataset
    self.download(name=name)
  
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
      tqdm().pandas()
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
  
  def prepare_data(self,folder='fall'):
    '''
    Create vectors based on bounding box center
    '''
    #location for Fall sequence bbox center vector
    fall_bbox = os.path.join(self.data_dir,'fall_bbox')
    if not os.path.exists(fall_bbox):
      os.mkdir(fall_bbox)

    #location for ADL sequence bbox center vector
    adl_bbox = os.path.join(self.data_dir,'adl_bbox')
    if not os.path.exists(adl_bbox):
      os.mkdir(adl_bbox)

    if folder == 'fall':
      dir = self.fall_dir
      name = 'fall_'
      bbox_file = fall_bbox
    else:
      dir = self.adl_dir
      name = 'adl_'
      bbox_file = adl_bbox
    #load model for bbox generation
    model = fasterrcnn(pretrained=True)
    if cuda.is_available():
      model = model.cuda()
    model.eval()
    l = os.listdir(dir)
    tqdm().pandas()
    for i in tqdm(range(len(l))):
      temp = []
      start = time.time()
      for j in sorted(os.listdir(os.path.join(dir,l[i]))):
        img_path = os.path.join(dir,l[i],j)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        #convert image to tensor for model
        tfm_img = transforms.ToTensor()(image)
        if cuda.is_available():
          tfm_img = tfm_img.cuda()
        x = [tfm_img]
        predictions = model(x)
        labels = predictions[0]['labels'].cpu().detach().numpy()
        person_box_coord = np.where(labels==1)[0]
        if person_box_coord.size > 1:
          id = np.argmax([predictions[0]['scores'][i].cpu().detach().numpy() for i in person_box_coord])
          id = person_box_coord[id]
        else:
          id = person_box_coord[0]
        box_coord = predictions[0]['boxes'][id].unsqueeze(0).cpu().detach().numpy().astype('int')
        for x1,y1,x2,y2 in box_coord:
          box_center = (int(x1+(x2-x1)/2),int(y1+(y2-y1)/2))
          temp.append(box_center)
      print('time take for a file: ',time.time() - start)
      cv2.imwrite(os.path.join(bbox_file, name+f'{i}.jpg'),np.array(temp))
