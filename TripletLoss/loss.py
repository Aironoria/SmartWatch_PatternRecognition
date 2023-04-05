import torch.nn as nn
import  torch
import  torch.nn.functional as F
class TripletLoss(nn.Module):
    def __init__(self,margin = 1):
        super(TripletLoss,self).__init__()
        self.margin = margin


    # minimize positive distance and maximum negative distance
    def forward(self,anchor,positive,negative):
        positive_distance = calc_distance(anchor,positive)
        negative_distance = calc_distance(anchor,negative)
        losses = F.relu(self.margin + positive_distance  - negative_distance)
        return losses.mean()

def calc_distance(x1,x2):
    # return cos_similarity(x1,x2)
    return euclid_distance(x1,x2)
def cos_similarity(x1,x2):
    return F.cosine_similarity(x1,x2)

def euclid_distance(x1,x2):
    # return (x1-x2).pow(2).sum(1)
    dist = nn.PairwiseDistance(p=2)
    return dist(x1,x2)
    # return  torch.clamp(torch.exp(-dist(x1,x2)), min=1e-7, max=1.0 - 1e-7)