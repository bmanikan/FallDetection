from dataset import *
from torch.utils.data import random_split,DataLoader

data = PrepareData('urfd',download=True,create_vectors=True)
print(data.labels)

labels = data.labels
temp = URFDDataset(labels)
train_size = int(0.8*len(temp))
test_size = len(temp) - train_size
train, test = random_split(temp, [train_size,test_size])
train_loader = DataLoader(train,batch_size=2,shuffle=True)
test_loader = DataLoader(train,batch_size=2,shuffle=False)



