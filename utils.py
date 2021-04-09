import cv2
import numpy as np
import os
import torch
from torch import cuda
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn as fasterrcnn
import time
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

def safedirs(path):
  if not os.path.exists(path):
    os.makedirs(path)

import cv2
import numpy as np
import os
import torch
from torch import cuda
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn as fasterrcnn
import time
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

def get_bbox(images):
  assert torch.is_tensor(images),"images should be of Tensor"
  #print(images.shape)
  model = fasterrcnn(pretrained=True)
  if cuda.is_available():
    images = images.cuda()
    model.cuda()
  model.eval()
  bbox = []
  for img in images:
    with torch.no_grad():
      preds = model([img])
    lbl = preds[0]['labels'].cpu().detach().numpy()
    #model predicts '1' label for person
    person_box_coord = np.where(lbl==1)[0]
    #if many boxes are available, then we choose one with high score
    if person_box_coord.size > 1:
      id = np.argmax([preds[0]['scores'][i].cpu().detach().numpy() for i in person_box_coord])
      id = person_box_coord[id]
    #if there is no person in the frame
    elif person_box_coord.size < 1:
      continue
    else:
      id = person_box_coord[0]
    #get the bbox coordinates
    box_coord = preds[0]['boxes'][id].unsqueeze(0).cpu().detach().numpy().astype('int')
    #calculate bbox center point
    for x1,y1,x2,y2 in box_coord:
      box_center = (int(x1+(x2-x1)/2),int(y1+(y2-y1)/2))
      bbox.append(box_center)
  #print(bbox)
  return bbox
    

def transform_image(images,labels):
  background = []
  lbls = []
  for image,label in zip(images,labels):
    bkg = np.zeros((640,480),np.uint8)
    start = time.time()
    points = get_bbox(image)
    print('time ',time.time() - start)
    for p in range(len(points)-1):
      bkg = cv2.line(bkg,points[p],points[p+1],(255,0,0),5)
    bkg = transforms.ToTensor()(bkg)
    lbl = 1 if 1 in label else 0
    background.append(bkg)
    lbls.append(lbl)
  #print(lbls)
  return torch.stack(background,dim=0), torch.IntTensor(lbls)