import time,os
import shutil
from download_dataset import urfd
import pandas as pd
import numpy as np
from label_enrichment import enrich_labels
from tqdm import tqdm
from utils import *
from torch.utils.data import random_split
from dataset_creation import createDataset
from embed_data import embedded_data
from model import create_model
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall

## Download dataset to the system
root = os.getcwd()
datapath = root + '/dataset/urfd'
logger.info(f'Dateset is downloaded in path {datapath}')
t= t_set()
urfd(datapath) 
logger.info(f'Total time for downloading and extracting entire dataset is {t_diff(t)}\n {"="*200}')
    

## Process the dataset
# create dataframe for further processing using the CSV file data
csv = root + '/dataset/urfd/csv'
df_fall = pd.read_csv('/content/dataset/urfd/csv/urfall-cam0-falls.csv',header=None)
df_adl = pd.read_csv('/content/dataset/urfd/csv/urfall-cam0-adls.csv',header=None)
dfs = pd.concat([df_fall,df_adl],ignore_index=True)
dfs = dfs[[0,1,2]]
dfs['path'] = None
logger_t.info(f"Enrich labels into the filenames extractend from the CSV files")
t = t_set()
enrich_labels(dfs,'/content/dataset/urfd/frame')
logger.info(f'Elapsed time for modifying filenames: {t_diff(t)}s.')


## Create subset of data for processing

#120 image dataset
#shutil.rmtree('/content/subset')

logger_t.info(f'{"="*100} \n\nSTEP 3: Create subset of 60/120/180 files of 50 Frames each from the Dataset')

size = 180
logger.info(f'Size of the subset: {size}')
ids = dfs[0].unique()[:60]
train_size = int(0.8*len(ids))
val_size = int(0.8*(len(ids) - train_size))
test_size = len(ids) - (train_size + val_size)
train_ids,valid_ids,test_ids = random_split(ids,[train_size,val_size,test_size],generator=torch.Generator().manual_seed(1))
logger_t.info(f'train_size={train_size}\nvalidation_size={val_size}\ntest_size={test_size}')
logger_t.info(f'train_ids={train_ids}\nvalidation ids={valid_ids}\ntest_ids={test_ids}')

train_destPath = os.path.join(os.getcwd(),'subset/train')
valid_destPath = os.path.join(os.getcwd(),'subset/valid')
test_destPath = os.path.join(os.getcwd(),'subset/test')
logger_t.info(f'train path= {train_destPath}\nvalidation path= {valid_destPath}\nTest path= {test_destPath}')

t = t_set()
createDataset(dfs,'/content/dataset/urfd/frame',train_destPath,ids = list(train_ids),size=size)
logger.info(f"Time elapsed for creating Train subset is {t_diff(t)}")
t = t_set()
createDataset(dfs,'/content/dataset/urfd/frame',valid_destPath,ids = list(valid_ids),size=size)
logger.info(f"Time elapsed for creating Validation subset is {t_diff(t)}")
t = t_set()
createDataset(dfs,'/content/dataset/urfd/frame',test_destPath,ids = list(test_ids),size=size)
logger.info(f"Time elapsed for creating Test subset is {t_diff(t)}")


## Create BBOX and Trajectory files

logger_t.info(f'{"="*100} \n\nSTEP 4: Create Trajectory images\n')
t = t_set()
root = os.getcwd()
embedded_data(root + '/subset')
logger.info(f'Total time taken to create trajectory images: {t_diff(t)}')

# Create loaders
logger_t.info(f'{"="*100} \n\nSTEP 5: Create Data generators\n')
datagen = tf.keras.preprocessing.image.ImageDataGenerator()
batch_size = 32
logger.info(f'batch size: {batch_size}')
root = os.getcwd()
train = datagen.flow_from_directory(root + '/trajectories/train',target_size=(224, 224),color_mode='grayscale',batch_size=batch_size)
valid = datagen.flow_from_directory(root + '/trajectories/valid',target_size=(224, 224),color_mode='grayscale',batch_size=batch_size)
test = datagen.flow_from_directory(root + '/trajectories/test',target_size=(224, 224),color_mode='grayscale',batch_size=batch_size)

model = create_model()
model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3), 
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy', Precision(), Recall()])
logger.info(f'{model.summary()}')

history = model.fit_generator(generator=train,
                              validation_data=valid,
                              epochs=10,
                              verbose=1)

logger.info(f'\nEvaluation Results')
model.evaluate(test)