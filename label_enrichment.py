def process_label(label,filename):
  if filename[:4] == 'fall' and label == 1:
      return 1
  else:
      return 0

def rename_file(df,root, filenames):
  for filename in filenames:
    ext = filename.split('.')[-1]
    file_id,frame_id = filename[:7], int(filename.split('.')[0].split('-')[-1])
    df_label = int(df[(df[0] == file_id) & (df[1] == frame_id)][2])
    label = process_label(df_label,file_id)
    name = f'{file_id}_{frame_id}_{label}.{ext}'
    new_path = os.path.join(root, name)
    old_path = os.path.join(root,filename)
    os.rename(old_path,new_path)


def enrich_labels(df, dataPath, folder=True):
  '''
  Embed labels into the data filenames
  df: data frame consisting of details on labels
  datapath: the path of the folder containing frames
  folder: indicates the structure of the data
  Returns: None
  '''
  if folder:
    root, dirs, _ = next(os.walk(dataPath))
    dir_array = [os.path.join(root,dir) for dir in dirs]
    for path in dir_array:
      root, _, files = next(os.walk(path))
      rename_file(df,root,files)
  else:
    root, _, files = next(os.walk(dataPath))
    rename_file(df, root, files)