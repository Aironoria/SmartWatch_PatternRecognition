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



class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))  # 初始化混淆矩阵，元素都为0
        self.num_classes = num_classes  # 类别数量，本例数据集类别为5
        self.labels = labels  # 类别标签

    def update(self, preds, labels):
        for p, t in zip(preds, labels):  # pred为预测结果，labels为真实标签
            self.matrix[p, t] += 1  # 根据预测结果和真实标签的值统计数量，在混淆矩阵相应位置+120_10

    def get_acc(self):
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # 混淆矩阵对角线的元素之和，也就是分类正确的数量
        return  sum_TP / n  # 总体准确率


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

def pt_to_ptl(model_name):
    model = torch.load(model_name +".pt")
    model.eval()
    scripted_module = torch.jit.script(model)
    optimize_for_mobile(scripted_module)._save_for_lite_interpreter(model_name+  ".ptl")

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
            data['gx']=data['gx']*10+4
            data['gy']= data['gy']*10 +6
            data['gz']= data['gz'] *10+8
            data.to_csv(save_dir+ "/"+ gesture + "."+filename,index_label="timestamp")

def edgeimpulse_to_csv(root):
    save_dir = root+"_converted"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for file in os.listdir(root):
        gesture = file.split(".")[0]
        save_file_parent = os.path.join(save_dir,gesture)
        if not os.path.exists(save_file_parent):
            os.mkdir(save_file_parent)
        save_file = os.path.join(save_file_parent,str(len(os.listdir(save_file_parent))) +".csv")
        with open(os.path.join(root,file)) as f:
            df = json.load(f)['payload']['values']
            df = pd.DataFrame(df)
            if(len(df) < 400):
                print(file + " only has " + str(len(df)) + " lines")
            if(len(df) >500):
                continue # unsplited
            df = df.loc[0:399]
            df.to_csv(save_file,index= False)

def split_data(root):
    save_dir = root + "_splited"
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    for gesture in os.listdir(root):
        # if not os.path.exists(os.path.join(save_dir, gesture)):
        #     os.mkdir(os.path.join(save_dir, gesture))
        for dir in os.listdir(os.path.join(root,gesture)):
            if dir == ".DS_Store":
                continue
            gyro = pd.read_csv(os.path.join(root,gesture,dir,"ACC.csv"))
            acc = pd.read_csv(os.path.join(root,gesture,dir,"GYRO.csv"))
            for i in range(3):
                save_file = os.path.join(save_dir,gesture,dir+"_"+str(i))
                os.makedirs(save_file)
                gyro_df = gyro.loc[i*400:i*400 +399]
                acc_df = acc.loc[i*400:i*400 +399]
                gyro_df.to_csv(save_file +"/ACC.csv",index=False)
                acc_df.to_csv(save_file +"/GYRO.csv",index=False)
    return save_dir

def random_sample_n(root,num):
    unselected = random.sample(os.listdir(root+"/Nothing"), len(os.listdir(root+"/Nothing")) - num)
    for file in unselected:
        os.remove(os.path.join(root, "data_25/Nothing", file))
    return root

def augment(root):
    for gesture in os.listdir(root):
        save_dir = os.path.join(root+"_augmented",gesture)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if gesture == "Nugget" or gesture == "Hamburg":
            for file in os.listdir(os.path.join(root,gesture)):
                df = pd.read_csv(os.path.join(root,gesture,file))
                for i in [0,5,10,15,20]:
                    df.loc[i : 379 + i ].to_csv(os.path.join(save_dir,str(len(os.listdir(save_dir)))+".csv"), index=False)
        else:
            for file in os.listdir(os.path.join(root,gesture)):
                df = pd.read_csv(os.path.join(root, gesture, file))
                df.loc[10 :390-1].to_csv(os.path.join(save_dir,str(len(os.listdir(save_dir)))+".csv"),index=False)
    return root+"_augmented"

def two_to_one_csv(root):
    for gesture in os.listdir(root):
        save_dir = os.path.join(root + "_merged", gesture)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for dir in os.listdir(os.path.join(root, gesture)):
            acc = pd.read_csv(os.path.join(root,gesture,dir,"ACC.csv"))
            gyro = pd.read_csv(os.path.join(root,gesture,dir,"GYRO.csv"))
            df = pd.concat([acc,gyro],axis=1)
            df.to_csv(save_dir +"/"+dir+".csv",index=False)
    return root+"_merged"


def get_n_nothing_from_content(nums):
    save_dir =str(nums)+"_Nothing"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    shutil.copytree("content/Nothing",save_dir+"/Nothing")
    splited_data = split_data(save_dir)
    merged_splited_data = two_to_one_csv(splited_data)
    sampled_data = random_sample_n(merged_splited_data,nums)
    shutil.rmtree(save_dir)
    shutil.rmtree(splited_data)
    os.rename(sampled_data,save_dir)

def convert_Click_data(root):
    splited_data = split_data(root)
    merged_splited_data = two_to_one_csv(splited_data)
    shutil.rmtree(root)
    shutil.rmtree(splited_data)
    os.rename(merged_splited_data,root)

def print_dir_len(dir):
    total = 0
    for gesture in os.listdir(dir):
        length =len(os.listdir(os.path.join(dir,gesture)))
        total += length
        print(gesture + str(length))
    print("Total: " + str(total))
    print()


def split_and_augment(root):
    train_ratio = 0.8
    split_train_test(root,train_ratio)
    train_dir= augment(root +"_train")
    test_dir = augment(root+"_test")
    dst = root + "_augmented_" + str((int) (train_ratio *100)) +"%"
    os.mkdir(dst)
    shutil.move(train_dir, dst+ "/train")
    shutil.move(test_dir, dst +"/test")
    shutil.rmtree(root + "_train")
    shutil.rmtree(root+"_test")




def split_train_test(root,train_ratio):
    for gesture in os.listdir(root):
        list = os.listdir(os.path.join(root,gesture))
        random.shuffle(list)
        train_size = int(len(list) * train_ratio)
        train_list = list[: train_size]
        test_list = list[train_size:]

        train_dir = os.path.join(root+ "_train",gesture)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        test_dir = os.path.join(root+ "_test",gesture)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        for file in train_list:
            shutil.copy(os.path.join(root,gesture,file), os.path.join(train_dir,file))
        for file in test_list:
            shutil.copy(os.path.join(root,gesture,file), os.path.join(test_dir,file))


def plot_fft(file_path,title):
    df=pd.read_csv(file_path)
    df_ax = fft.fft(df['az'])


    x = np.arange(95)

    abs_y = np.abs(df_ax[5:])  # 取复数的绝对值，即复数的模(双边频谱)
    angle_y = np.angle(df_ax)  # 取复数的角度

    plt.figure()
    plt.plot(x, abs_y)
    plt.title(title)

    # plt.figure()
    # plt.plot(x, angle_y)
    # plt.title('phase')
    plt.show()

def plot_dir_fft(dir,title):
    for file in os.listdir(dir):
        plot_fft(os.path.join(dir,file),title)

if __name__ == '__main__':

    # plot_dir("10-3-new")
    # plot_fft("palm/circle_air_index_palm/04_28_12_59_57.csv")
    # plot_fft("palm/circle_table_index_palm/04_28_13_00_28.csv")
    plot_dir_fft("assets/palm/left_air_index_palm", "air")
    plot_dir_fft("assets/palm/left_table_index_palm", "table")
    # convert_to_edgeimpulse("palm")
    pass







