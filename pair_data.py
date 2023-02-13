import os
import random

from torchvision.transforms import transforms

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



class PairDataset(Dataset):
    def __init__(self, labels,datalen,path_list, transform=None):
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


    def __len__(self):
        return self.dataset_len

    def load_for_rnn(self,path,rnn_len,total_len):
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

    def load_for_cnn(self,path):
        total_len=100
        item = pd.read_csv(path.strip())
        start_index = random.randint(20, 30)
        item = item.iloc[start_index:start_index + total_len].values

        item = torch.tensor(item).to(torch.float32)

        item = torch.reshape(item.T, (6, 10, -1))
        if self.transform:
            item = self.transform(item)
        return item

    def __getitem__(self, idx):

        path1 = random.choice(self.path_list)
        label1 = path1.split(os.sep)[-2]

        path2 = random.choice(self.path_list)
        label2 = path2.split(os.sep)[-2]
        if idx%2 ==0:
            while label1 != label2:
                path2 = random.choice(self.path_list)
                label2 = path2.split(os.sep)[-2]
        item1 = self.load_for_cnn(path1)
        item2 = self.load_for_cnn(path2)
        label = 1 if label2 == label1 else 0

        # item = self.load_for_rnn(path,5,100)

        return item1,item2, torch.from_numpy(np.array([label],dtype=np.float32))


    def get_label_dict(self):
        res ={}
        for i in range (len(self.labels)):
            res[i]=self.labels[i]
        return res


class PairTestDataset(Dataset):
    def __init__(self, labels,support_path,test_path, transform=None):
        self.size = len(test_path)
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

    def load_for_cnn(self, path):
        total_len = 100
        item = pd.read_csv(path.strip())
        start_index = random.randint(20, 30)
        item = item.iloc[start_index:start_index + total_len].values

        item = torch.tensor(item).to(torch.float32)

        item = torch.reshape(item.T, (6, 10, -1))
        if self.transform:
            item = self.transform(item)
        return item

    def __len__(self):
        return self.size
        pass

    def __getitem__(self, idx):
        path = self.test_path[idx]
        label = path.split(os.sep)[-2]
        label = torch.tensor(self.labels.index(label))

        target = self.load_for_cnn(path)

        support =[]
        for i in self.support_path:
            i_label  =torch.tensor(self.labels.index( i.split(os.sep)[-2]))
            support.append((self.load_for_cnn(i),i_label))
        return target,label,support


def load_dataset(root,mode,participant=None):
    train_list=[]
    support_list=[]
    test_list=[]
    lables =[category for category in os.listdir(os.path.join(root, os.listdir(root)[0]))]

    if mode ==CROSSPERSON_20:
        train_list,support_list,test_list = get_cross_n_list(0.2,root,participant)
    elif mode ==CROSSPERSON_05:
        train_list,support_list,test_list = get_cross_n_list(0.05,root,participant)
    elif mode ==CROSSPERSON_10:
        train_list,support_list,test_list = get_cross_n_list(0.1,root,participant)
    else:
        print("Data: mode error")

    return PairDataset(lables,10000,train_list),PairDataset(lables,1000,train_list), PairTestDataset(lables,support_list,test_list)



def get_cross_n_list(ratio,root,participant):
    train_list=[]
    support_list=[]
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

    support_list += [os.path.join(root,participant,item) for item in train]
    test_list += [os.path.join(root,participant,item) for item in test]
    return  train_list,support_list,test_list

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
    train,validate,test = load_dataset("assets/input/ten_data_",CROSSPERSON_20,"cxy")
    for i in range(20):
        a = train[i]
        # b = train[1]
    print()
    pass
