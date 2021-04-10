
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # layers of a CNN
        self.conv1 = nn.Conv2d(1,32,3,stride=2,padding=1)   
        self.conv2 = nn.Conv2d(32,64,3,stride=2,padding=1)
        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        
        # Linear layers
        self.fc1 = nn.Linear(128*7*7, 1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,2)
        
        # pooling layers
        self.pool = nn.MaxPool2d(2,2)
        
        # dropout layers
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # forward behavior
        x = F.relu(self.conv1(x))     # 112x112
        x = self.pool(x)              # 56x56
        x = F.relu(self.conv2(x))     # 28x28
        x = self.pool(x)              # 14x14
        x = F.relu(self.conv3(x))     # 14x14
        x = self.pool(x)              # 7x7

        
        x = x.view(-1,128*7*7)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x