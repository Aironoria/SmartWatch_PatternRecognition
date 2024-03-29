import os

import torch.fft
import torch.nn as nn
import torch.nn.functional as F

import config


#10 6*8
#120_10 26*36
#2 21 *20
#5 11 *12


class TAPID_CNNEmbedding(nn.Module):
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
        super(TAPID_CNNEmbedding,self).__init__()
        net =[]
        conv_depth = 3
        for i in range(conv_depth):
            net.extend(self.conv_block(6 if i==0 else 2**(i+4) , 2**(i+5)))
        self.convs = nn.Sequential(*net)
        self.average_pool = nn.AvgPool1d(kernel_size=3,stride=1)
        #2:22  3:9  4:3
        if config.embedding_size ==64:
            if conv_depth ==4:
                self.shape = 256*2
            elif conv_depth ==3:
                self.shape = 128*6
        elif config.embedding_size ==80:
            if conv_depth ==4:
                self.shape = 256*3
            elif conv_depth ==3:
                self.shape = 128*8
        elif config.embedding_size ==100:
            if conv_depth ==4:
                self.shape = 256*4
            elif conv_depth ==3:
                self.shape = 128*10
        elif config.embedding_size ==128:
            if conv_depth ==4:
                self.shape = 256*6
            elif conv_depth ==3:
                self.shape = 128*14
        self.linear1 = nn.Linear(self.shape,config.embedding_size)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        y=x
        x =self.convs(x)
        x = self.average_pool(x)
        x = x.view(-1, self.shape)
        x=self.linear1(x)
        # return x
        return self.sigmoid(x)

class CNNEmbeddingNet(nn.Module):
    def __init__(self):
        super(CNNEmbeddingNet,self).__init__()
        self.shape_1 = 15 * 6 * 6
        self.conv1 = nn.Conv2d(6, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 15, kernel_size=3)
        self.fc1 = nn.Linear(self.shape_1, 1000)
        self.sigmoid =nn.Sigmoid()
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.shape_1)
        x = self.sigmoid(self.fc1(x))
        return x
class SiameseCNN(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseCNN,self).__init__()
        self.embedding_net = embedding_net
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(128,1)
        print("cnn siamese init")
    def fc_distance(self,x1,x2):
        x = torch.abs(x1 - x2)
        x = self.fc(x)
        return self.sigmoid(x)
    def forward(self,x1,x2):
        x1 =  self.embedding_net(x1)
        x2 =  self.embedding_net(x2)
        return self.fc_distance(x1,x2)
        # return cosine_similarity(x1,x2)
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