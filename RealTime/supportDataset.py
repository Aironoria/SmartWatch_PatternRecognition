import os
import random

from torchvision.transforms import transforms

import config

import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
class SupportDataset(Dataset):
    def __init__(self,path_list):

        mean =[0.88370824, -1.0719419, 9.571041, -0.0018323545, -0.0061315685, -0.0150832655]
        std =[0.32794556, 0.38917893, 0.35336846, 0.099675156, 0.117989756, 0.06230596]
        self.transform = transforms.Compose([
            transforms.Normalize(
                mean, std
            )
        ])
        # self.labels = self.get_labels(root)
        self.labels =['scroll_down', 'click', 'scroll_up', 'spread', 'swipe_right', 'pinch', 'swipe_left', 'touchdown', 'nothing', 'touchup']

        self.path_list=path_list
        self.labels = [i for i in self.labels if i not in config.ignored_label]
        ignored_path = []
        for i in self.path_list:
            for j in config.ignored_label:
                if j in i:
                    ignored_path.append(i)
        self.path_list = [i for i in self.path_list if i not in ignored_path]
        print("dataset init: path list length: " + str(len(self.path_list)))


    def __len__(self):
        return len(self.path_list)

    def load_for_cnn(self, path):
        total_len = 128
        item = pd.read_csv(path.strip())
        if len(item) == total_len:
            start_index=0
        else:
            start_index = random.randint(20, 30)
        item = item.iloc[start_index:start_index + total_len].values

        item = torch.tensor(item).to(torch.float32)

        item = torch.reshape(item.T, (6, 2, -1))

        if self.transform:
            item = self.transform(item)
        item = torch.reshape(item, (6,-1))
        return item
    def __getitem__(self, index):

        path = self.path_list[index]
        path = path.replace("/",os.sep)
        label = path.split(os.sep)[-2]
        label =torch.tensor(self.labels.index(label))
        item = self.load_for_cnn(path)

        return item ,label
    def get_label_dict(self):
        res ={}
        for i in range (len(self.labels)):
            res[i]=self.labels[i].replace("scroll","swipe")
        return res

def load_support_dataset(root, surface, length,support_include_all_conditions = False):
    include_new=False
    gestures=[]
    support_list = []
    support_surface =[]
    if support_include_all_conditions:
        for dir in os.listdir(root):
            if dir != "new":
                support_surface.append(dir)
        if surface =="new" or include_new:
            support_surface.append("new")
    else:
        support_surface = [surface]
        if surface == "new":
            support_surface.append("base")
    for i,the_surface in enumerate(support_surface):
        df = []
        for gesture in os.listdir(os.path.join(root,the_surface)):
            for filename in os.listdir(os.path.join(root,the_surface,gesture)):
                df.append({
                    "path": os.path.join(root, the_surface,gesture, filename),
                    "gesture": gesture
                })

        df = pd.DataFrame(df)
        support_list.extend(df.groupby("gesture").head(length)['path'])

    return SupportDataset( support_list)