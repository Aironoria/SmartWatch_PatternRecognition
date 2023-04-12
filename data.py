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
CROSSPERSON_20 ="crossperson_20"
CROSSPERSON_05 ="crossperson_05"
CROSSPERSON_10 ="crossperson_10"


class FoodDataset(Dataset):
    def __init__(self, root,path_list, transform=None):

        mean =[0.88370824, -1.0719419, 9.571041, -0.0018323545, -0.0061315685, -0.0150832655]
        std =[0.32794556, 0.38917893, 0.35336846, 0.099675156, 0.117989756, 0.06230596]
        # self.labels = self.get_labels(root)
        self.labels =['scroll_down', 'click', 'scroll_up', 'spread', 'swipe_right', 'pinch', 'swipe_left', 'touchdown', 'nothing', 'touchup']
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


    def __len__(self):
        return len(self.path_list)

    def load_for_rnn(self, path):
        rnn_len = 5
        total_len = 100
        item = pd.read_csv(path.strip())
        start_index = random.randint(20, 30)
        item = item.iloc[start_index:start_index + total_len].values
        item = torch.tensor(item)
        item = item.to(torch.float32)

        a = item.numpy()
        b = torch.reshape(item.T, (6, rnn_len, -1)).numpy()
        item = torch.reshape(item.T, (6, rnn_len, -1))
        if self.transform:
            item = self.transform(item)
            c = item.numpy()

        item = torch.reshape(item, (6, -1)).T
        d = item.numpy()
        item = torch.reshape(item, (((int)(total_len / rnn_len)), -1))
        e = item.numpy()
        return item

    def load_for_cnn(self, path):
        total_len = 100
        item = pd.read_csv(path.strip())
        start_index = random.randint(20, 30)
        item = item.iloc[start_index:start_index + total_len].values

        item = torch.tensor(item).to(torch.float32)

        item = torch.reshape(item.T, (6, 10, -1))

        if self.transform:
            item = self.transform(item)
        item = torch.reshape(item, (6,-1))
        return item

    def __getitem__(self, index):

        path = self.path_list[index]
        path = path.replace("/",os.sep)
        label = path.split(os.sep)[-2]
        label =torch.tensor(self.labels.index(label))
        if config.network == 'cnn':
            item = self.load_for_cnn(path)
        elif config.network =='rnn':
            item =self.load_for_rnn(path)
        else:
            print('config net error')
        return item ,label

    def get_labels(self,root):
        root = os.path.join(root, os.listdir(root)[0])
        return [category for category in os.listdir(root) if os.path.isdir(os.path.join(root, category))]
    def get_label_dict(self):
        res ={}
        for i in range (len(self.labels)):
            res[i]=self.labels[i]
        return res


def load_test_dataset(root,surface):
    test = []
    with open(os.path.join(root + "_train_test", surface, "test.txt"), 'r') as f:
        for line in f.readlines():
            test.append(line)
    test_list = [os.path.join(root, surface, filename) for filename in test]

    root = os.path.join(root, surface)
    return FoodDataset(root, test_list)


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
            with open(os.path.join(root + "_train_test", person, "train.txt"), 'r') as f:
                for line in f.readlines():
                    train_list.append(os.path.join(root, person, line))
            with open(os.path.join(root + "_train_test", person, "test.txt"), 'r') as f:
                for line in f.readlines():
                    test_list.append(os.path.join(root, person, line))
    elif mode == CROSSPERSON:
        for person in os.listdir(root):
            for gesture in os.listdir(os.path.join(root,person)):
                for filename in os.listdir(os.path.join(root,person,gesture)):
                    path = os.path.join(root,person,gesture,filename)
                    if person== participant:
                        test_list.append(path)
                    else:
                        train_list.append(path)
    elif mode ==CROSSPERSON_20:
        train_list,test_list = get_cross_n_list(20,root,participant)
    elif mode ==CROSSPERSON_05:
        train_list,test_list = get_cross_n_list(5,root,participant)
    elif mode ==CROSSPERSON_10:
        train_list,test_list = get_cross_n_list(10,root,participant)
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
    ratio = ratio*0.01
    train_list=[]
    test_list=[]
    for person in os.listdir(root):
        for gesture in os.listdir(os.path.join(root, person)):
            for filename in os.listdir(os.path.join(root, person, gesture)):
                path = os.path.join(root, person, gesture, filename)
                if person != participant:
                    train_list.append(path)
    gestures=[]
    filelist=[]
    with open(os.path.join(root + "_train_test", participant,"all.txt"),'r') as f:
        for line in f.readlines():
            gesture = line.split(os.sep)[0]
            if gesture not in gestures:
                gestures.append(gesture)
                filelist.append([])
            idx = gestures.index(gesture)
            filelist[idx].append(line)

    train = []
    test=[]
    for item in filelist:
        length = int(len(item)*ratio)
        train+=item[:length]
        test +=item[length:]
    [os.path.join(root,participant,item) for item in train],[os.path.join(root,participant,item) for item in test]

    train_list += [os.path.join(root,participant,item) for item in train]
    test_list += [os.path.join(root,participant,item) for item in test]
    return  train_list,test_list

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



if __name__ == "__main__":

    # a = load_dataset(os.path.join("../assets", "input", "10-27_11-15_12-04_len65_sampled"))[0]
    # # getStat(a)
    # for label in a.labels:
    #     print('"'+label+'"',end=",")
    # train,test =  load_dataset("content")
    train,test =  load_dataset("assets/input/ten_data_",INPERSON,"cxy",50)
    pass
