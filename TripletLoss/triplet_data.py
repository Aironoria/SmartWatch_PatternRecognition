import os
import random

import Augmentation
import config

from torchvision.transforms import transforms
import segmentation

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
torch.set_printoptions(precision=4,sci_mode=False)
OVERALL ="overall"
INPERSON = "inperson"
CROSSPERSON = "crossperson"
CROSSPERSON_20 ="crossperson_20"
CROSSPERSON_05 ="crossperson_05"
CROSSPERSON_10 ="crossperson_10"


class TripletDataset(Dataset):
    def __init__(self, labels,datalen,path_list, network ="cnn",transform=None):
        self.dataset_len = datalen

        self.labels = labels
        self.path_list=path_list
        self.length = len(self.path_list)
        self.time_domain =True
        self.transform = transform

        mean = [0.88370824, -1.0719419, 9.571041, -0.0018323545, -0.0061315685, -0.0150832655]
        std = [0.32794556, 0.38917893, 0.35336846, 0.099675156, 0.117989756, 0.06230596]
        self.transform = transforms.Compose([
        transforms.Normalize(
          mean,std
        )
        ])
        self.network = network
        self.labels = [i for i in self.labels if i not in config.ignored_label]
        ignored_path = []
        for i in self.path_list:
            for j in config.ignored_label:
                if j in i:
                    ignored_path.append(i)
        self.path_list = [i for i in self.path_list if i not in ignored_path]
        print(f"Use Jitter: {config.use_Jitter}, Use TimeWarp: {config.use_Time_warp}, Use MagWarp: {config.use_Mag_warp}")

        self.start_point_calculater = StartPointCalculater("../segmentation_result_peak.csv")
    def __len__(self):
        return self.dataset_len

    def load_item(self,path,warping_different_axis=False):
        total_len = config.embedding_size

        item = pd.read_csv(path.strip())
        #use path as key
        short_path ="/".join(path.split("/")[4:]).strip()
        start_index = self.start_point_calculater.get_start_point(short_path,total_len, random.randint(-5, 5))


        item = item.iloc[start_index:start_index + total_len].values

        if config.use_Time_warp:
            if warping_different_axis:
                item = Augmentation.TimeWarping(item,sigma=1.2,same_for_axis=False)
            else:
                if random.random() > 0.5:
                    item = Augmentation.TimeWarping(item)
        if config.use_Mag_warp:
            if random.random() > 0.5:
                item = Augmentation.MagnitudeWarping(item)
        if config.use_Jitter:
            if random.random() > 0.5:
                item = Augmentation.Jitter(item)
        item = torch.tensor(item).to(torch.float32)

        item = torch.reshape(item.T, (6, 2, -1))
        if self.transform:
            item = self.transform(item)
        item = torch.reshape(item, (6, -1))
        return item
    def __getitem__(self, idx):

        anchor_path = random.choice(self.path_list)
        anchor_label = anchor_path.split(os.sep)[-2]
        anchor = self.load_item(anchor_path)

        positive_path = random.choice(self.path_list)
        while positive_path.split(os.sep)[-2] != anchor_label or anchor_path == positive_path:
            positive_path = random.choice(self.path_list)
        positive = self.load_item(positive_path)


        # negative_path = random.choice(self.path_list)
        # if negative_path.split(os.sep)[-2] == anchor_label:
        #     negative = self.load_item(negative_path, warping_different_axis=True)
        # else:
        #     negative = self.load_item(negative_path)

        negative_path = random.choice(self.path_list)
        while negative_path.split(os.sep)[-2] == anchor_label:
            negative_path = random.choice(self.path_list)
        negative = self.load_item(negative_path)

        return anchor,positive,negative


    def get_label_dict(self):
        res ={}
        for i in range (len(self.labels)):
            res[i]=self.labels[i]
        return res


class PairTestDataset(Dataset):
    def __init__(self, labels,support_path,test_path,network="cnn", transform=None):

        mean = [0.88370824, -1.0719419, 9.571041, -0.0018323545, -0.0061315685, -0.0150832655]
        std = [0.32794556, 0.38917893, 0.35336846, 0.099675156, 0.117989756, 0.06230596]
        self.transform = transforms.Compose([
            transforms.Normalize(
                mean, std
            )
        ])
        self.support_path =support_path
        self.test_path = test_path
        self.labels = labels
        self.network = network
        self.start_points = pd.read_csv("../segmentation_result.csv")

        self.labels = [i for i in self.labels if i not in config.ignored_label]
        ignored_path = []
        for i in self.support_path:
            for j in config.ignored_label:
                if j in i:
                    ignored_path.append(i)
        self.support_path = [i for i in self.support_path if i not in ignored_path]
        ignored_path = []
        for i in self.test_path:
            for j in config.ignored_label:
                if j in i:
                    ignored_path.append(i)
        self.test_path = [i for i in self.test_path if i not in ignored_path]
        self.size = len(self.test_path)
        self.start_point_calculater = StartPointCalculater("../segmentation_result_peak.csv")

    def get_label_dict(self):
        res = {}
        for i in range(len(self.labels)):
            res[i] = self.labels[i]
        return res

    def __len__(self):
        return self.size
        pass
    def load_data(self,path):
        total_len = config.embedding_size
        item = pd.read_csv(path.strip())
        short_path = "/".join(path.split("/")[4:]).strip()
        start_index = self.start_point_calculater.get_start_point(short_path,total_len, 0)

        item = item.iloc[start_index:start_index + total_len].values

        item = torch.tensor(item).to(torch.float32)

        item = torch.reshape(item.T, (6, 2, -1))
        if self.transform:
            item = self.transform(item)
        item = torch.reshape(item, (6, -1))
        return item
    def __getitem__(self, idx):
        path = self.test_path[idx]
        label = path.split(os.sep)[-2]
        label = torch.tensor(self.labels.index(label))

        target = self.load_data(path)

        support =[]
        for i in self.support_path:
            i_label  =torch.tensor(self.labels.index( i.split(os.sep)[-2]))
            support.append((self.load_data(i),i_label))
        return target,label,support


def load_dataset(root,mode,participant=None,network="cnn",n=0):
    train_list=[]
    support_list=[]
    test_list=[]
    lables =[category for category in os.listdir(os.path.join(root, os.listdir(root)[0]))]

    if mode == CROSSPERSON:
        train_list,support_list,test_list = get_cross_n_list(root,participant,n)
    elif mode == INPERSON:
        train_list,support_list,test_list = get_in_person_list(root,participant)
    elif mode ==OVERALL:
        train_list,support_list,test_list = load_overall_dataset(root)
    else:
        print("Data: mode error")
    return TripletDataset(lables,config.siamese_train_size,train_list,network),TripletDataset(lables,config.siamese_test_size,train_list,network), PairTestDataset(lables,support_list,test_list,network)

def get_in_person_list(root,participant):
    train_list=[]
    support_list =[]
    query_list =[]
    with open(os.path.join(root + "_train_test", participant, "train.txt"), 'r') as f:
        for line in f.readlines():
            train_list.append(os.path.join(root, participant, line))
    df=[]
    with open(os.path.join(root + "_train_test", participant, "test.txt"), 'r') as f:
        for line in f.readlines():
            df.append({
                "gesture": line.split(os.sep)[0],
                "path": os.path.join(root, participant, line)
            })
    df = pd.DataFrame(df)
    for name,group in df.groupby("gesture"):
        support_list+= group.head(5)["path"].to_list()
        query_list += group.iloc[5:]["path"].to_list()

    return train_list, support_list, query_list
def load_overall_dataset(root):
    train_list = []
    support_list = []
    query_list = []
    for person in os.listdir(root):
        if person == ".DS_Store":
            continue
        with open(os.path.join(root + "_train_test", person, "train.txt"), 'r') as f:
            for line in f.readlines():
                train_list.append(os.path.join(root, person, line))
        df = []
        with open(os.path.join(root + "_train_test", person, "test.txt"), 'r') as f:
            for line in f.readlines():
                df.append({
                    "gesture": line.split(os.sep)[0],
                    "path": os.path.join(root, person, line)
                })
        df = pd.DataFrame(df)
        for name, group in df.groupby("gesture"):
            support_list += group.head(5)["path"].to_list()
            query_list += group.iloc[5:]["path"].to_list()

    return train_list, support_list, query_list


def get_cross_n_list(root,participant,n=0):
    train_list = []
    support_list = []
    query_list=[]
    for person in os.listdir(root):
        if person == participant or person == ".DS_Store":
            continue

        with open(os.path.join(root + "_train_test", person, "train.txt"), 'r') as f:
            for line in f.readlines():
                train_list.append(os.path.join(root, person, line))

    df = []

    with open(os.path.join(root + "_train_test", participant, "train.txt"), 'r') as f:
        for line in f.readlines():
            df.append({
                "gesture": line.split(os.sep)[0],
                "path": os.path.join(root, participant, line)
            })
    df = pd.DataFrame(df)
    train = df.groupby("gesture").head(n)["path"]
    train_list += df.groupby("gesture").head(n)["path"].to_list()

    df = []
    with open(os.path.join(root + "_train_test", participant, "test.txt"), 'r') as f:
        for line in f.readlines():
            df.append({
                "gesture": line.split(os.sep)[0],
                "path": os.path.join(root, participant, line)
            })
    df = pd.DataFrame(df)
    for name, group in df.groupby("gesture"):
        support_list += group.head(5)["path"].to_list()
        query_list += group.iloc[5:]["path"].to_list()
    return train_list, support_list, query_list


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


    train,validate,test = load_dataset("../assets/input/ten_data_",OVERALL)
    print(train[0][0])
    pass



