import torch.nn as nn
import  torch
import  torch.nn.functional as F
class ContrastiveLoss(nn.Module):
    def __init__(self,margin):
        super(ContrastiveLoss,self).__init__()
        self.margin = margin
        self.eps = 1e-9
    # minimize distance if target == 0 (same)
    def forward(self,distance,target):
        # losses = 0.5 * (target.float() * distance.pow(2) +
        #                 (1 + -1*target).float() *F.relu(self.margin - (distance + self.eps)).pow(2))
        losses = 0.5 * ((1 + -1 * target).float() * distance.pow(2) +
                         target.float()* F.relu(self.margin - (distance + self.eps)).pow(2))
        a = losses.mean()
        b = losses.sum()
        return a