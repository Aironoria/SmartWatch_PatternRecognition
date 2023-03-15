import os

import torch.fft
import torch.nn as nn
import torch.nn.functional as F
#10 6*8
#120_10 26*36
#2 21 *20
#5 11 *12

class oneDCNN(nn.Module):
    def conv_block(self, in_channel, out_channel):
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
        # self.conv1 = self.conv_block(6,32)
        # self.conv2 = self.conv_block(32,64)
        # self.conv3 = self.conv_block(64,128)
        # self.conv4 = self.conv_block(128,256)
        # self.conv5 = self.conv_block(256,512)
        net =[]
        for i in range(5):
            net.extend(self.conv_block(6 if i==0 else 2**(i+4) , 2**(i+5)))
        self.convs = nn.Sequential(*net)
        self.average_pool = nn.AvgPool1d(kernel_size=3,stride=1)
        self.linear = nn.Linear(512*3,128)
        self.fcout=nn.Linear(128,10)
    def forward(self,x):
        x =self.convs(x)
        # x = self.average_pool(x)
        x = x.view(-1, 512*3)
        x=self.linear(x)
        x=F.relu(x)
        x=self.fcout(x)
        return F.softmax(x,dim=1)
class Net(nn.Module):

    def __init__(self,output_num):
        #     15 *3 *3
        self.shape_1 = 15 *6 *6
        super(Net,self).__init__()

        self.conv1 = nn.Conv2d(6,10,kernel_size=3)
        self.conv2 = nn.Conv2d(10,15,kernel_size=3)
        self.fc1 =  nn.Linear(self.shape_1, 1000)
        self.fc2 = nn.Linear(1000,output_num)
        print('cnn init')

    def forward(self,x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x= x.view(-1, self.shape_1)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return F.softmax(x,dim=1)



class SiameseNet(nn.Module):
    def __init__(self,output_num):
        #     15 *3 *3
        self.shape_1 = 15 *6 *6
        # self.shape_1 = 30 *4*4
        super(SiameseNet,self).__init__()

        self.conv1 = nn.Conv2d(6,10,kernel_size=3)
        self.conv2 = nn.Conv2d(10,15,kernel_size=3)
        self.conv3 = nn.Conv2d(15,30,kernel_size=3)
        self.fc1 =  nn.Linear(self.shape_1, 1000)
        self.fcOut = nn.Linear(1000,1)
        self.fcOu2 = nn.Linear(50, 1)
        self.sigmoid =nn.Sigmoid()
        print("cnn siamese init")
    def convs(self,x):
        x =  F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x= F.relu(self.conv3(x))
        return x

    def abs_distance(self,x1,x2):
        x = torch.abs(x1 - x2)
        x = self.fcOut(x)
        return self.sigmoid(x)
    def forward(self,x1,x2):
        a =x1
        b=x2
        x1 = self.convs(x1)
        x1 = x1.view(-1, self.shape_1)
        x1 =self.sigmoid(self.fc1(x1))

        x2 = self.convs(x2)
        x2 = x2.view(-1, self.shape_1)
        x2 = self.sigmoid(self.fc1(x2))

        # return cosine_similarity(x1,x2)
        return normal_form(x1,x2)

class Siamese_RNN(nn.Module):
    def __init__(self,output_num):
        super(Siamese_RNN,self).__init__()
        input_size=6*5
        hidden_size = 64*4
        n_layer =1
        a =100
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layer, batch_first=True)
        self.fc1 = nn.Linear(hidden_size,a)
        self.sigmoid = nn.Sigmoid()
        self.fcOut = nn.Linear(hidden_size, 1)
        self.fc2 =nn.Linear (a,output_num)
        self.fc3 = nn.Linear(output_num,1)
    def forward(self,x1,x2):
        out, (h_n,c_n) = self.lstm(x1)
        x1 = h_n[-1]
        x1  = self.sigmoid(self.fc1(x1))
        out, (h_n, c_n) = self.lstm(x2)
        x2 = h_n[-1]
        x2 = self.sigmoid(self.fc1(x2))
        x =torch.abs(x1-x2)
        # x = torch.cat((x1,x2,torch.abs(x1-x2),(x1+x2)/2,x1*x2),1)
        # # x=torch.exp(-torch.sum(torch.abs(x1 - x2), dim=1,keepdim=True))
        x = self.fc2(x)
        x= self.fc3(x)
        # x = (x1-x2).pow(2).sum(1,keepdims=True).sqrt()
        # return x
        return self.sigmoid(x)

class RNN(nn.Module):
    def __init__(self,output_num):
        super(RNN,self).__init__()
        input_size=6*5
        hidden_size = 64
        n_layer =1

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layer, batch_first=True)
        self.fc1 = nn.Linear(hidden_size,output_num)
        print('rnn init')

    def forward(self,x):
        out, (h_n,c_n) = self.lstm(x)
        x = out[:,-1:,:].squeeze(1)     #batch_size , seq_len, hidden_size
        x = self.fc1(x)
        return F.softmax(x,dim=1)

def normal_form(x1,x2,order=2):
    return torch.unsqueeze( nn.functional.pairwise_distance(x1,x2,2),1)
def cosine_similarity(x1,x2):
    return torch.unsqueeze(nn.functional.cosine_similarity(x1, x2),1)

def similarity_score(input1, input2):
    # Get similarity predictions:
    dif = input1.squeeze() - input2.squeeze()
    norm = torch.norm(dif, p=1, dim=dif.dim() - 1)
    y_hat = torch.exp(-norm)
    y_hat = torch.clamp(y_hat, min=1e-7, max=1.0 - 1e-7)
    return y_hat