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


class RNN(nn.Module):
    def __init__(self,output_num):
        super(RNN,self).__init__()
        input_size=6*5
        hidden_size = 64
        n_layer =1

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layer, batch_first=True)
        self.fc1 = nn.Linear(hidden_size,output_num)

    def forward(self,x):
        out, (h_n,c_n) = self.lstm(x)
        x = out[:,-1:,:].squeeze(1)     #batch_size , seq_len, hidden_size
        x = self.fc1(x)
        return F.softmax(x,dim=1)
