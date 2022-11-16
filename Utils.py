import os.path
import json
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.ticker import MultipleLocator
from torch.utils.mobile_optimizer import optimize_for_mobile
import shutil
from    numpy import fft

import cnn


class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))  # 初始化混淆矩阵，元素都为0
        self.num_classes = num_classes  # 类别数量，本例数据集类别为5
        self.labels = labels  # 类别标签

    def update(self, preds, labels):
        for p, t in zip(preds, labels):  # pred为预测结果，labels为真实标签
            self.matrix[p
            , t] += 1  # 根据预测结果和真实标签的值统计数量，在混淆矩阵相应位置+120_10

    def get_acc(self):
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # 混淆矩阵对角线的元素之和，也就是分类正确的数量
        return  sum_TP / n  # 总体准确率

    def set_matrix(self):
        pass

    def plot(self,root,tittle,save):  # 绘制混淆矩阵
        matrix = self.matrix
        # print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix (acc=' + str(self.get_acc()) + ')')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(root, tittle),bbox_inches = 'tight')
        else:
            plt.show()
        plt.clf()




def plot_loss(save_dir,train_loss, train_acc , test_loss, test_acc):
    plt.plot(np.arange(len(train_loss)), train_loss, label="train loss.jpg")
    plt.plot(np.arange(len(test_loss)), test_loss, label="valid loss.jpg")

    plt.title('loss.jpg')
    plt.legend()  # 显示图例
    plt.savefig(os.path.join(save_dir, "loss.jpg"),bbox_inches = 'tight')
    # plt.show()
    plt.clf()
    plt.plot(np.arange(len(train_acc)), train_acc, label="train acc.jpg")

    plt.plot(np.arange(len(test_acc)), test_acc, label="valid acc.jpg")
    plt.legend()  # 显示图例
    plt.xlabel('epoches')
    # plt.ylabel("epoch")
    plt.title('acc.jpg')
    plt.savefig(os.path.join(save_dir, "acc.jpg"),bbox_inches = 'tight')
    # plt.show()
    plt.clf()


def plot_data(data,save_dir,label,file):

    locator = 150
    index = range(1, len(data['ax']) + 1)

    plt.rcParams["figure.figsize"] = (20, 10)
    plt.subplot(2,1,1)

    acc = data[["ax","ay","az"]]
    gyro = data[["gx","gy","gz"]]
    plt.plot(index, acc['ax'], label='ax', linestyle='solid', marker=',')
    plt.plot(index, acc['ay'], label='ay', linestyle='solid', marker=',')
    plt.plot(index, acc['az'], label='az', linestyle='solid', marker=',')
    plt.gca().xaxis.set_major_locator(MultipleLocator(locator))
    plt.title(label)
    plt.xlabel("Sample #")
    plt.ylabel("Acceleration (G)")
    plt.legend()


    plt.subplot(2,1,2)
    plt.plot(index, gyro['gx'], label='gx', linestyle='solid', marker=',')
    plt.plot(index, gyro['gy'], label='gy', linestyle='solid', marker=',')
    plt.plot(index, gyro['gz'], label='gz', linestyle='solid', marker=',')
    plt.gca().xaxis.set_major_locator(MultipleLocator(locator))
    plt.xlabel("Sample #")
    plt.ylabel("Gyroscope (deg/sec)")
    plt.legend()
    # plt.show()
    dir = "pic_" + save_dir
    path = os.path.join(dir,label)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(dir+ "/"+label+"/"+file.split(".")[0])
    plt.clf()




def pth_to_pt():
    model = torch.load("model.pth")
    model.eval()
    input = torch.rand(1,6,15,16)
    torch.jit.trace(model,input).save("model.pt")

def pt_to_ptl(path):
    model = torch.load(path)
    model.eval()
    scripted_module = torch.jit.script(model)
    optimize_for_mobile(scripted_module)._save_for_lite_interpreter(path+  "l")

def plot_dir(dir):
    for gesture in os.listdir( dir):
        for file in os.listdir(os.path.join(dir,gesture)):
            data = pd.read_csv(os.path.join(dir,gesture,file))
            plot_data(data,dir,gesture, file)

def convert_to_edgeimpulse(root):
    save_dir = root +"_edge"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for gesture in os.listdir(root):
        dir = os.path.join(root,gesture)
        for filename in os.listdir(dir):
            data = pd.read_csv(dir + "/" + filename)
            if transform:
                print(filename)
                data['gx'] = data['gx'] *2 + 4
                data['gy'] = data['gy'] *2 + 6
                data['gz'] = data['gz'] *2 + 8
            data.to_csv(save_dir+ "/"+ gesture.split("_")[0] + "."+filename,index_label="timestamp")

def edgeimpulse_to_csv(root):
    save_dir = root+"_converted"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for file in os.listdir(root):
        gesture = file.split(".")[0]
        save_file_parent = os.path.join(save_dir,gesture)
        if not os.path.exists(save_file_parent):
            os.mkdir(save_file_parent)
        filename = file.split(".")[1]
        if file.split(".")[-2].startswith("s"):
            filename += "_"+file.split(".")[-2]
        save_file = os.path.join(save_file_parent,filename +".csv")
        with open(os.path.join(root,file)) as f:
            df = json.load(f)['payload']['values']
            df = pd.DataFrame(df)
            if gesture == "touchup":
                file_length = 60
            else:
                file_length =70
            file_length = 70
            if(len(df) < file_length):
                print(file + " only has " + str(len(df)) + " lines")
            df = df.loc[0:file_length-1]

            if transform:
                # df[3] = (df[3] -4)/2
                # df[4] = (df[4] -6)/2
                # df[5] = (df[5] -8)/2
                df[3] = (df[3])/50
                df[4] = (df[4])/50
                df[5] = (df[5])/50

            df.to_csv(save_file,index= False)

def random_sample_n(root,num):
    unselected = random.sample(os.listdir(root+"/Nothing"), len(os.listdir(root+"/Nothing")) - num)
    for file in unselected:
        os.remove(os.path.join(root, "data_25/Nothing", file))
    return root



def print_dir_len(dir):
    total = 0
    for gesture in os.listdir(dir):
        length =len(os.listdir(os.path.join(dir,gesture)))
        total += length
        print(gesture + str(length))
    print("Total: " + str(total))
    print()


#using window size and sliding to get sample from each file
def augment(root,window_size,sliding_step,subs =[""]):
    save_root = root+"_augmented"
    for sub in subs:
        if not sub == "":
            dir = os.path.join(root,sub)
        else:
            dir = root
        for category in os.listdir(dir):
            # if category=="touchup":
            #     sliding_step=1
            # else:
            #     sliding_step=2
            save_dir = os.path.join(save_root,sub, category)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for file in os.listdir(os.path.join(dir,category)):
                a=os.path.join(dir,category,file)
                df = pd.read_csv(os.path.join(dir,category,file))
                start_index = 0
                while True:
                    if (len(df) < start_index + window_size ):
                        break
                    df.loc[start_index : start_index+window_size -1 ].to_csv(os.path.join(save_dir,file.split(".")[0]+"_"+str(start_index)+".csv"), index=False)
                    start_index +=sliding_step
    return save_root


def split_train_test(root,train_ratio):
    save_dir = root+"_"
    for gesture in os.listdir(root):
        list = os.listdir(os.path.join(root,gesture))
        random.shuffle(list)
        train_size = int(len(list) * train_ratio)
        train_list = list[: train_size]
        test_list = list[train_size:]

        train_dir = os.path.join(save_dir, "train",gesture)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        test_dir = os.path.join(save_dir,"test",gesture)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        for file in train_list:
            shutil.copy(os.path.join(root,gesture,file), os.path.join(train_dir,file))
        for file in test_list:
            shutil.copy(os.path.join(root,gesture,file), os.path.join(test_dir,file))
    return save_dir

def plot_fft(file_path,title):
    df=pd.read_csv(file_path)
    df_ax = fft.fft(df['az'])


    x = np.arange(100)

    abs_y = np.abs(df_ax[5:])  # 取复数的绝对值，即复数的模(双边频谱)
    angle_y = np.angle(df_ax)  # 取复数的角度
    psd = df_ax * np.conj(df_ax) / 100

    plt.figure()
    plt.plot(x, psd)
    plt.title(title)

    # plt.figure()
    # plt.plot(x, angle_y)
    # plt.title('phase')
    plt.show()

def plot_dir_fft(dir,title):
    for file in os.listdir(dir):
        plot_fft(os.path.join(dir,file),title)


def split_nothing(dir):
    augment(dir, 70, 64)

def sample_from_dir(root,train_size = 1760, test_size = 440):
    save_dir = root+"_sampled1"
    shutil.copytree(root,save_dir)
    # train 1760 test 440
    for gesture in os.listdir(os.path.join(save_dir,"train")):
        path = os.path.join(save_dir,"train",gesture)
        unselected = random.sample(os.listdir(path), max(len(os.listdir(path)) - train_size,0) )
        for file in unselected:
            os.remove(os.path.join(path, file))
    for gesture in os.listdir(os.path.join(save_dir,"test")):
        path = os.path.join(save_dir,"test",gesture)
        unselected = random.sample(os.listdir(path), max(len(os.listdir(path)) - test_size,0))
        for file in unselected:
            os.remove(os.path.join(path, file))
    pass

if __name__ == '__main__':
    transform =True
    dir = "assets/input/11-15_len(49)_with10-27"
    # sample_from_file("assets/aa", 150,40)
    # split_train_test("assets/input/test/testing_converted",0.7)

    # dir =  split_file_and_train_test("assets/10-12")
    # sample_from_file("assets/input/test/testing_converted_",50,10,subs=["train","test"])
    # sample_from_file("assets/input/test/testing_converted",50,10)

    # print(cnn.Net())
    # convert("Poh_")
    # label = ["Apple","Burger","Edamame","Noodle","Nugget","Peanut","Rice" ]
    # confusion = ConfusionMatrix(num_classes=len(label), labels=label)

    # pt_to_ptl("assets/res/200epochs_3720/10-12_augmented.pt")
    # convert_to_edgeimpulse("assets/input/test/edge")
    # edgeimpulse_to_csv("assets/input/test/testing")
    # convert_to_edgeimpulse(dir)
    # edgeimpulse_to_csv(dir)
    # sample_from_file(dir +"_converted_",50,2,subs=["train","test"])
    # split_train_test(dir, 0.8)
    # sample_from_file(dir +"_",49,2,subs=["train","test"])
    # print_dir_len(dir+"_/train")
    # print_dir_len(dir+"__augmented/train")
    # print_dir_len(dir)
    # edgeimpulse_to_csv(dir)
    # split_train_test(dir,0.8)
    # augment(dir+"_",49,1,["train","test"])
    sample_from_dir(dir,2500,600)
    print_dir_len(dir+"_sampled1/train")

    print_dir_len(dir+"_sampled1/test")
    pass









