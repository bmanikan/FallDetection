import urllib.request as urlreq
from zipfile import ZipFile
import shutil
from utils import safedirs
import os


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
  # raise assertion if the url is not of file type
  assert ext in map.keys(), "invalid url"
  name = url.split('/')[-1]
  filePath = os.path.join(root, map[ext])
  safedirs(filePath)
  filename = os.path.join(filePath,name)
  if os.path.exists(filename):
    return
  urlreq.urlretrieve(url, filename)
  #if the file downloaded is zip file then extract original files
  if ext == 'zip':
    extract_zip(filename,root,map)
    

def urfd(dataPath,num,clean=False):
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
    download(f,dataPath)
  #download frame zip files from the project website
  for n in tqdm(range(1,num)):
    # Add zeros to adapt files in Website
    if n < 10:
      n = '0'+str(n)
    #only 31 Fall sequences available
    if int(n)<31:
      fall_url = f'http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-{n}-cam0-rgb.zip'
      download(fall_url,datapath)
    adl_url = f'http://fenix.univ.rzeszow.pl/~mkepski/ds/data/adl-{n}-cam0-rgb.zip'
    download(adl_url,dataPath)
    if int(n) == num-1 and clean:
      cleanup(dataPath)

