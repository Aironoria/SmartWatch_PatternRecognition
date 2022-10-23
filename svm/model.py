import os

import torch.fft
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_num, output_num):
        super(Net,self).__init__()
        self.shape1 = input_num *6
        self.fc1 =  nn.Linear(input_num *6, 1000)
        self.fc2 = nn.Linear(1000, output_num)

    def forward(self,x):
        x = x.view(-1,self.shape1)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        x= F.relu(x)
        return F.softmax(x,dim=1)

