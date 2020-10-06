import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from generate_path import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        n1, n2, n3 = 1000, 1000, 1000
        self.fc1 = nn.Linear(steps+1, n1)
        self.fc2 = nn.Linear(n1, n2)
        self.fc3 = nn.Linear(n2, n3)
        self.fc4 = nn.Linear(n3, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = F.relu(self.fc1(x)) # ReLU: max(x, 0)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return self.sigmoid(x)
