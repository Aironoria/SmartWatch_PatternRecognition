import torch.nn as nn
import  torch
import  torch.nn.functional as F
class Classification_Net(nn.Module):
    def __init__(self,class_num,embedding_net):
        super(Classification_Net,self).__init__()
        self.embedding= embedding_net
        self.layer = nn.Linear(128,class_num)
    def forward(self,x):
        with torch.no_grad():
            x = self.embedding(x)
        x = self.layers(x)
        return F.softmax(x,-1)
