import os
import random

from torchvision.transforms import transforms

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
torch.set_printoptions(precision=4,sci_mode=False)



class FoodDataset(Dataset):
    def __init__(self, root, split = False,train=True, transform=None):

        mean =[0.88370824, -1.0719419, 9.571041, -0.0018323545, -0.0061315685, -0.0150832655]
        std =[0.32794556, 0.38917893, 0.35336846, 0.099675156, 0.117989756, 0.06230596]
        self.labels = [category for category in os.listdir(root) if os.path.isdir(os.path.join(root, category))]
        self.path_list = self.get_data_list(root)
        self.length = len(self.path_list)
        self.time_domain =True
        self.transform = transform
        if  self.time_domain:
            self.transform = transforms.Compose([
            transforms.Normalize(
              mean,std
            )
            ])


    def __len__(self):
        return len(self.path_list)


    def __getitem__(self, index):

        a = index // len(self.path_list)
        index= index  % len(self.path_list)
        path = self.path_list[index]
        label = path.split(os.sep)[-2]
        label =torch.tensor(self.labels.index(label))

        item = pd.read_csv(path)
        total_len =60
        item =item.iloc[0:total_len].values
        if not self.time_domain:
            item = np.abs(np.fft.fftn(item))
        # print(path)
        item = torch.tensor(item)
        item =item.to(torch.float32)

        a= item.numpy()
        b = torch.reshape(item.T,(6,2,-1)).numpy()
        item = torch.reshape(item.T,(6,2,-1))
        if self.transform:
            item = self.transform(item)
            c=item.numpy()
        rnn_len =4
        item = torch.reshape(item,(6,-1)).T
        d =item.numpy()
        item = torch.reshape(item,(total_len/rnn_len,-1))
        e=item.numpy()

        use_gyro =True
        # use_gyro =False

        if not use_gyro:
            item = item.split(3, 0)[0]

        return item ,label

    def get_data_list(self, root):
        res = list()
        for category in self.labels:
            for file in os.listdir(os.path.join(root, category)):
                res.append(os.path.join(root, category, file))
        return res
    def get_label_dict(self):
        res ={}
        for i in range (len(self.labels)):
            res[i]=self.labels[i]
        return res


def load_dataset(root):
    return FoodDataset(os.path.join(root,"train")), FoodDataset(os.path.join(root,"test"))

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

    a = load_dataset(os.path.join("assets","input","12-04_sampled_without_swipeup"))[0]
    # getStat(a)
    for label in a.labels:
        print('"'+label+'"',end=",")


    # train,test =  load_dataset("content")

