import os

import torch.fft
import torch.nn as nn
import torch.nn.functional as F



class oneDCNN(nn.Module):
    def conv_block(self, in_channel, out_channel,convs = 2):
        return [
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, padding=1, kernel_size=3),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channel, out_channels=out_channel, padding=1, kernel_size=3),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        ]
    def __init__(self):
        super(oneDCNN,self).__init__()

        net =[]
        for i in range(3):
            net.extend(self.conv_block(6 if i==0 else 2**(i+4) , 2**(i+5)))
        self.convs = nn.Sequential(*net)
        self.average_pool = nn.AvgPool1d(kernel_size=3,stride=1)
        # self.linear = nn.Linear(512*2,128)
        # self.fcout=nn.Linear(128,10)
        self.linear = nn.Linear(128*6,64)
        self.fcout=nn.Linear(64,5)
    def forward(self,x):
        x =self.convs(x)
        x = self.average_pool(x)
        x = x.view(-1, 128*6)
        embedding=self.linear(x)
        x=F.relu(embedding)
        x=self.fcout(x)
        return embedding, F.softmax(x,dim=1)