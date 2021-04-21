import torch
from torchvision import transforms
from tqdm.notebook import tqdm
from skimage.metrics import structural_similarity
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time,os
from PIL import Image
from torchvision import transforms


def safedirs(path):
  if not os.path.exists(path):
    os.makedirs(path)


def person_detector(image):
  '''
  Moving object detection by frame differencing
  input: torch tensor object of shape [batch,50,3,480,640] and labels array
  output: tuple of torch tensor [batch,1,224,224] and label object 
  '''
  temp_img = image.numpy().transpose(0,2,3,1)
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
    center = (int((x+w)/2), int((y+h)/2))
    center_points.append(center)
  return center_points


def get_transform():
  transform = transforms.Compose([transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, ), (0.5, ))])
  return transform
    

def image_embedding(images,labels):
  transform = get_transform()
  background = []
  lbls = []
  for image,label in zip(images,labels):
    lbl=[0,0]
    bkg = np.zeros((480,640),np.uint8)
    start = time.time()
    points = person_detector(image)
    #print('time ',time.time() - start)
    for p in range(len(points)-1):
      bkg = cv2.line(bkg,points[p],points[p+1],(255,0,0),5)
    bkg = transform(Image.fromarray(bkg))
    lbl[1] = 1 if 1 in label else 0
    lbl[0] = 1 if 1 not in label else 0
    background.append(bkg)
    lbls.append(lbl)
  #print(lbls)
  return torch.stack(background,dim=0), torch.FloatTensor(lbls)

