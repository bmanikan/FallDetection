import cv2
import os
import time


def process_video(videoPath, destPath, folder=False):
  '''
  Extract frames from the given video and store in designated path
  videoPath: path of video
  destPath: path to save frames
  folder: Whether to create seperate folder for each video in destination
  Returns None.
  '''
  cap = cv2.VideoCapture(videoPath)
  fps = cap.get(cv2.CAP_PROP_FPS)
  width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  filename = videoPath.split('/')[-1].split('.')[0]
  if folder:
    destPath = os.path.join(destPath, filename)
    safedirs(destPath)
  while (cap.isOpened()):
    ret,frame = cap.read()
    if ret == False:
      break
    timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    file_path = os.path.join(destPath, filename+f'_{timestamp}.jpg')
    cv2.imwrite(file_path, frame)
  cap.release()
  return None


def extract(dataPath,destPath,folder=False):
  '''
  Extract frames from each video file and store in the destination
  dataPath = folder path of video files
  destPath = folder path to store processed frames 
  folder: whether to create seperate folder for each video file in destination
  '''
  root, _, videos = next(os.walk(dataPath))
  assert len(videos) > 0, "Folder doesn't contain any files"
  for video in videos:
    if video.split('.')[-1] not in ['mp4', 'avi']:
      continue
    abs_path = os.path.join(root,video)
    extract_frames(abs_path, destPath, folder=folder)
  return None




