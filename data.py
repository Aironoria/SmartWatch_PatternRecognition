import os
import random

from torchvision import transforms
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
torch.set_printoptions(precision=4,sci_mode=False)



class FoodDataset(Dataset):
    def __init__(self, root, split = False,train=True, transform=None):

        mean =[1.57083, 1.5717753, 1.5739009, 1.5739968, 1.5713093, 1.5705085]
        std =[4.0307536, 4.0310283, 4.0326715, 4.0330863, 4.0325427, 4.032277]
        self.labels = [category for category in os.listdir(root) if os.path.isdir(os.path.join(root, category))]
        self.path_list = self.get_data_list(root)
        self.length = len(self.path_list)
        self.transform = transform
        # self.transform = transforms.Compose([
        # transforms.Normalize(
        #   mean,std
        # )
        # ])


    def __len__(self):
        return len(self.path_list)


    def __getitem__(self, index):
        a = index // len(self.path_list)
        index= index  % len(self.path_list)
        path = self.path_list[index]
        label = path.split("/")[-2]
        label =torch.tensor(self.labels.index(label))

        item = pd.read_csv(path)
        item = np.abs(np.fft.fftn(item))
        item = torch.tensor(item).to(torch.float32)

        item = torch.reshape(item.T,(1,-1))
        if self.transform:
            item = self.transform(item)

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

    a = load_dataset("assets/10-12_augmented")[0]
    # getStat(a)
    # print(a.labels)
    print(a[0])

    # train,test =  load_dataset("content")

