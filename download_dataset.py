import urllib.request as urlreq
from zipfile import ZipFile
import shutil
import os
from tqdm.notebook import tqdm
import datetime
import time


t_set = lambda: datetime.datetime.now().astimezone()
t_diff = lambda t: str(t_set() - t)
t_stamp = lambda t=None: str(t) if t else str(t_set())

logger_t.info(f'{"="*100} \n\nSTEP 1: Download dataset into the system')

def safedirs(path):
  if not os.path.exists(path):
    os.makedirs(path)

def extract_zip(src, dest, map):
    '''
    Extracts the contents downloaded from the URL
    '''
    zip_ref = ZipFile(src,'r')
    for name in zip_ref.namelist():
      ext = name.split('.')[-1]
      if ext in map.keys():
        dest = os.path.join(dest,map[ext])
        break
    zip_ref.extractall(dest)
    logger_t.info(f'... into {dest}')
    zip_ref.close()

def cleanup(dataPath):
  '''
  clean the temp files
  '''
  for folder in os.listdir(dataPath):
    if folder == 'temp':
      cleanup_path = os.path.join(dataPath,'temp')
      shutil.rmtree(cleanup_path)
      break
  return

def download(url, root):
  '''
  downloads the file from the url
  and put it in desired folder
  '''
  
  map = {'csv':'csv', 'zip':'temp', 'png':'frame', 'jpg':'frame', 'mp4':'video', 'avi':'video'}
  ext = url.split('.')[-1] 
  # raise if the url is of not file type
  assert ext in map.keys(), "invalid url"
  name = url.split('/')[-1]
  filePath = os.path.join(root, map[ext])
  safedirs(filePath)
  filename = os.path.join(filePath,name)
  if os.path.exists(filename):
    return 
  logger_t.info(f"Downloading data files from URL:{url}")
  urlreq.urlretrieve(url, filename)
  #if the file downloaded is zip file then extract original files
  if ext == 'zip':
    extract_zip(filename,root,map)
    

def urfd(dataPath,num=40,clean=False):
  '''
  download the files from the URFD data url
  dataPath: destination path
  num: number of files to download
  clean: whether to clean the temp files
  '''
  #csv file url's
  csv = ['http://fenix.univ.rzeszow.pl/~mkepski/ds/data/urfall-cam0-adls.csv',
         'http://fenix.univ.rzeszow.pl/~mkepski/ds/data/urfall-cam0-falls.csv']
  #download csv files
  for f in csv:
    t = t_set()
    download(f,dataPath)
    logger.info(f'Elapsed time for downloading and extracting the file is {t_diff(t)}.\n')
  #download frame zip files from the project website
  for n in tqdm(range(1,num+1)):
    # Add zeros to adapt files in Website
    if n < 10:
      n = '0'+str(n)
    #only 31 Fall sequences available
    if int(n)<31:
      t = t_set()
      fall_url = f'http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-{n}-cam0-rgb.zip'
      download(fall_url,datapath)
      logger.info(f'Elapsed time for downloading and extracting the file is {t_diff(t)}\n')
    t=t_set()
    adl_url = f'http://fenix.univ.rzeszow.pl/~mkepski/ds/data/adl-{n}-cam0-rgb.zip'
    download(adl_url,dataPath)
    logger.info(f'Elapsed time for downloading and extracting the file is {t_diff(t)}\n')
    
    if n == num-1 and clean:
      cleanup(dataPath)



if __name__ = "__main__":
  root = os.getcwd()
  datapath = root + '/dataset/urfd'
  logger.info(f'Dateset is downloaded in path {datapath}')
  t= t_set()
  urfd(datapath) 
  logger.info(f'Total time for downloading and extracting entire dataset is {t_diff(t)}\n {"="*200}')
    