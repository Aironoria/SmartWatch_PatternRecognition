import os

import torch.fft
import torch.nn as nn
import torch.nn.functional as F
#10 6*8
#120_10 26*36
#2 21 *20
#5 11 *12



class Net(nn.Module):

    def __init__(self,output_num):
        #     15 *3 *3
        self.shape_1 = 15 *3 *3
        super(Net,self).__init__()

        self.conv1 = nn.Conv2d(6,10,kernel_size=3)
        self.conv2 = nn.Conv2d(10,15,kernel_size=3)
        self.fc1 =  nn.Linear(self.shape_1, 1000)
        self.fc2 = nn.Linear(1000,output_num)

    def forward(self,x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x= x.view(-1, self.shape_1)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return F.softmax(x,dim=1)
