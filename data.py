import os
import random

from torchvision.transforms import transforms

import config

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
torch.set_printoptions(precision=4,sci_mode=False)

OVERALL ="overall"
INPERSON = "inperson"
CROSSPERSON = "crossperson"


class FoodDataset(Dataset):
    def __init__(self, root,path_list, transform=None,network="cnn",labels=None):

        mean =[0.88370824, -1.0719419, 9.571041, -0.0018323545, -0.0061315685, -0.0150832655]
        std =[0.32794556, 0.38917893, 0.35336846, 0.099675156, 0.117989756, 0.06230596]
        # self.labels = self.get_labels(root)
        self.labels =['scroll_down', 'click', 'scroll_up', 'spread', 'swipe_right', 'pinch', 'swipe_left', 'touchdown', 'nothing', 'touchup']
        if labels is not None:
            self.labels = labels
        self.network=network
        self.path_list=path_list
        self.time_domain =True
        self.transform = transform
        if  self.time_domain:
            self.transform = transforms.Compose([
            transforms.Normalize(
              mean,std
            )
            ])

        self.labels = [i for i in self.labels if i not in config.ignored_label]
        ignored_path = []
        for i in self.path_list:
            for j in config.ignored_label:
                if j in i:
                    ignored_path.append(i)
        self.path_list = [i for i in self.path_list if i not in ignored_path]
        print("dataset init: path list length: " + str(len(self.path_list)))
        self.start_point_calculater = StartPointCalculater("../segmentation_result_peak.csv")


    def __len__(self):
        return len(self.path_list)


    def load_for_cnn(self, path):
        test =True
        total_len = 64
        item = pd.read_csv(path.strip())
        short_path = "/".join(path.split("/")[4:]).strip()
        start_index = self.start_point_calculater.get_start_point(short_path, total_len, 0 if test else random.randint(-5, 5))
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
        item  = self.load_for_cnn(path)

        return item ,label
    def get_label_dict(self):
        res ={}
        for i in range (len(self.labels)):
            res[i]=self.labels[i].replace("scroll","swipe")
        return res

def load_test_dataset(root,surface):
    test = []
    with open(os.path.join(root + "_train_test", surface, "test.txt"), 'r') as f:
        for line in f.readlines():
            test.append(line)
    test_list = [os.path.join(root, surface, filename) for filename in test]

    root = os.path.join(root, surface)
    return FoodDataset(root, test_list)


def load_support_dataset(root, surface, length,include_all_conditions=False):
    include_new = False
    support_surface= []
    support_list = []
    gestures = []
    if include_all_conditions:
        for dir in os.listdir(root):
            if dir != "new":
                support_surface.append(dir)
        if surface == "new" or include_new:
            support_surface.append("new")
    else:
        support_surface = [surface]
        if surface == "new":
            support_surface.append("base")
    for i, the_surface in enumerate(support_surface):
        df = []
        with open(os.path.join(root + "_train_test", the_surface, "train.txt"), 'r') as f:
            for line in f.readlines():
                gesture = line.split(os.sep)[0]
                gestures.append(gesture)
                df.append({
                    "path": os.path.join(root, the_surface, line),
                    "gesture": gesture
                })
        df = pd.DataFrame(df)
        support_list.extend(df.groupby("gesture").head(length)['path'])
    labels = sorted(list(set(gestures)))
    # labels = ['click', 'pinch', 'scroll_down', 'scroll_up', 'spread', 'swipe_left', 'swipe_left_vertical', 'swipe_right', 'swipe_right_vertical']
    return FoodDataset(root, support_list,labels=labels)

def load_query_dataset(root, surface):
    query_list = []
    gestures = []
    with open(os.path.join(root + "_train_test", surface, "test.txt"), 'r') as f:
        for line in f.readlines():
            gestures.append(line.split(os.sep)[0])
            query_list.append(os.path.join(root, surface, line))
    if surface == "new":
        with open(os.path.join(root + "_train_test", "base", "test.txt"), 'r') as f:
            for line in f.readlines():
                gestures.append(line.split(os.sep)[0])
                query_list.append(os.path.join(root, "base", line))
    labels = sorted(list(set(gestures)))
    # labels = ['click', 'pinch', 'scroll_down', 'scroll_up', 'spread', 'swipe_left', 'swipe_left_vertical','swipe_right', 'swipe_right_vertical']
    return FoodDataset(root, query_list, labels=labels)
def load_dataset(root,mode,participant=None,n=None):
    train_list=[]
    test_list=[]
    if mode == INPERSON:
        if n == None:
            with open(os.path.join(root+"_train_test",participant,"train.txt"),'r') as f:
                for line in f.readlines():
                    train_list.append(os.path.join(root,participant,line))
            with open(os.path.join(root+"_train_test",participant,"test.txt"),'r') as f:
                for line in f.readlines():
                    test_list.append(os.path.join(root,participant,line))
        else:
            train_list,test_list = get_inperson_n_list(n,root,participant)
    elif mode == OVERALL:
        for person in os.listdir(root):
            if person == ".DS_Store":
                continue
            with open(os.path.join(root + "_train_test", person, "train.txt"), 'r') as f:
                for line in f.readlines():
                    train_list.append(os.path.join(root, person, line))
            with open(os.path.join(root + "_train_test", person, "test.txt"), 'r') as f:
                for line in f.readlines():
                    test_list.append(os.path.join(root, person, line))
    elif mode == CROSSPERSON:
        train_list,test_list = get_cross_n_list(n,root,participant)
    else:
        print("Data: mode error")

    return FoodDataset(root,train_list), FoodDataset(root,test_list)


def get_inperson_n_list(ratio,root,participant):
    ratio = ratio*0.01
    test_list=[]
    # with open(os.path.join(root + "_train_test", participant, "train.txt"), 'r') as f:
    #     for line in f.readlines():
    #         train_list.append(os.path.join(root, participant, line))
    with open(os.path.join(root + "_train_test", participant, "test.txt"), 'r') as f:
        for line in f.readlines():
            test_list.append(os.path.join(root, participant, line))

    gestures = []
    filelist = []
    with open(os.path.join(root + "_train_test", participant, "train.txt"), 'r') as f:
        for line in f.readlines():
            gesture = line.split(os.sep)[0]
            if gesture not in gestures:
                gestures.append(gesture)
                filelist.append([])
            idx = gestures.index(gesture)
            filelist[idx].append(line)
    temp=[]
    for item in filelist:
        length = int(len(item)*ratio)
        temp+=item[:length]
    train_list = [os.path.join(root,participant,filename) for filename in temp]

    return  train_list,test_list

def get_cross_n_list(ratio,root,participant):
    train_list=[]
    test_list=[]
    for person in os.listdir(root):
        if person == participant:
            continue

        with open(os.path.join(root + "_train_test", participant, "train.txt"), 'r') as f:
            for line in f.readlines():
                train_list.append(os.path.join(root, participant, line))

    df = []
    test =[]
    with open(os.path.join(root + "_train_test", participant, "train.txt"), 'r') as f:
        for line in f.readlines():
            df.append({
                "gesture": line.split(os.sep)[0],
                "path": line
            })
    df = pd.DataFrame(df)
    with open(os.path.join(root + "_train_test", participant, "test.txt"), 'r') as f:
        for line in f.readlines():
            test.append(line)
    train = df.groupby("gesture").head(ratio)["path"]
    train_list += [os.path.join(root, participant, item) for item in train]
    test_list += [os.path.join(root, participant, item) for item in test]
    return train_list, test_list


def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(6)
    std = torch.zeros(6)
    data_max = torch.zeros(6)
    data_min = torch.zeros(6)
    count=0
    for X, _ in train_loader:
        print(count,end="\t")
        for d in range(6):
            a = X[:, d, :, :].mean()
            print(a ,end="\t")
            mean[d] += X[:, d,:,  :].mean()
            std[d] += X[:, d, :, :].std()
            data_max[d] = max( data_max[d], X[:, d,  :].max())
            data_min[d] = min(data_min[d], X[:, d,  :].min())
        print("")
        count+=1
    mean.div_(len(train_data))
    std.div_(len(train_data))
    print(list(mean.numpy()), list(std.numpy()))
    # print(list(data_max.numpy()), list (data_min.numpy()))
    return list(mean.numpy()), list(std.numpy())

class StartPointCalculater:
    def __init__(self,path):
        self.start_points = pd.read_csv(path)

    def get_start_point(self,short_path,total_len,offset):
        start_index = self.start_points[self.start_points["path"] == short_path]["peak"].values[0] - (int)(total_len / 2)
        start_index += offset
        # clamp
        if start_index < 0:
            start_index = 0
        if start_index + total_len > 199:
            start_index = 199 - total_len
        return start_index



if __name__ == "__main__":

    # a = load_dataset(os.path.join("../assets", "input", "10-27_11-15_12-04_len65_sampled"))[0]
    # # getStat(a)
    # for label in a.labels:
    #     print('"'+label+'"',end=",")
    # train,test =  load_dataset("content")
    train,test =  load_dataset("assets/input/ten_data_",INPERSON,"cxy",50)
    pass
