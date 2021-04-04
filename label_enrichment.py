import pandas as pd
import os

def assign_label(df,root,filenames):
  for filename in filenames:
    #processed files contain '_' hence we can omit those
    if '_' in filename:
      continue
    #get the extension from the file
    ext = filename.split('.')[-1]
    split_array = filename.split('.')[0].split('-')
    #extract id and frame
    file_id,frame_id = '-'.join(split_array[:2]), int(split_array[-1])
    #adl files doesnt contain falls hence assigning label 0 to all files
    if 'adl' in file_id:
      label = 0
    else:
      #check the label in dataframe for id and frame
      df_label = int(df[(df[0] == file_id) & (df[1] == frame_id)][2])
      label = 1 if df_label == 1 else 0
    #Rename the files
    name = f'{file_id}_{frame_id}_{label}.{ext}'
    new_path = os.path.join(root, name)
    old_path = os.path.join(root,filename)
    os.rename(old_path,new_path)


def enrich_labels(df,dataPath,folder=True):
  '''
  Embed labels into the data filenames
  df: data frame consisting of details on labels
  datapath: the path of the folder containing frames
  folder: indicates the structure of the data
  '''
  if folder:
    root, dirs, _ = next(os.walk(dataPath))
    dir_array = [os.path.join(root,dir) for dir in dirs]
    for path in dir_array:
      root, _, files = next(os.walk(path))
      assign_label(df,root,files)
  else:
    root, _, files = next(os.walk(dataPath))
    assign_label(df, root, files)