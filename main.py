from dataset import *
from torch.utils.data import random_split,DataLoader

data = PrepareData('urfd',download=True,create_vectors=True)
print(data.labels)

ids = df[0].unique()
train_size = int(0.8*len(ids))
test_size = len(ids) - train_size
train_ids,test_ids = random_split(ids,[train_size,test_size],generator=torch.Generator().manual_seed(43))
train = clipDataset(df,'/content/dataset/urfd/frame',ids = list(train_ids))
test = clipDataset(df,'/content/dataset/urfd/frame',ids = list(test_ids))
train_loader = DataLoader(train,batch_size=8,shuffle=True)
test_loader = DataLoader(test,batch_size=8,shuffle=True)
loaders = {'train':train_loader,'test':test_loader}



