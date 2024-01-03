#evaluating 1D cnn model using new surfaces data

import os
import time

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.mobile_optimizer import optimize_for_mobile
import numpy as np
import Utils
import cnn
import config
import data
import torch
from torch.utils.data import DataLoader

import pair_data
from TripletLoss import triplet_data
from Utils import ConfusionMatrix

import  network
device = torch.device('cpu')




def eval(net,test_loader,support_size,save_dir="",plot=True):
  title = f"conf_test_triplet_{support_size}.png"
  net.eval()
  class_indict = test_loader.dataset.get_label_dict()
  label = [label for _, label in class_indict.items()]
  print("confusion matrix", label)

  confusion = ConfusionMatrix(num_classes=len(label), labels=label)
  with torch.no_grad():
    for target,target_label,support_set in test_loader:
        data = target.to(device)
        labels = target_label.to(device)
        embedding, outputs = net(data)
        _, predicted = torch.max(outputs.data, 1)
        confusion.update(predicted.numpy(), labels.numpy())
  if plot:
    confusion.plot(save_dir ,title,save=True)
  return confusion.get_acc()



def get_save_root():
    # return os.path.join("assets", "res", "cnn_" + dataset + "_ignored_3gestures_" + str(N_epoch) + "1d")
    return  os.path.join("..","assets","res",  model_dir+"_"+dataset_dir+"_embedding_64")


def get_save_dir(participant,surface):
  root =get_save_root()
  res = os.path.join(root,participant,surface)
  if not os.path.exists(res):
    os.makedirs(res)
    print("create dir: " + res)

  return res




def eval_traditional_network(net_path,participant,support_size=1,support_include_all_conditions=False):
    net = network.oneDCNN()
    net.load_state_dict(torch.load(net_path))

    x = []
    y = []
    eval_num= 0
    start = time.time()
    print("evaluating triplet network, support size = " + str(support_size))
    print(f"mode is {net_path}")

    # surfaces = os.listdir(os.path.join("assets", "input", dataset_dir))
    # surfaces = ['base','lap','wall','left','new']

    for surface in surfaces:
    # for surface in ["new"]:
        print("eval surface: " + surface +f"save dir is {get_save_dir(participant,surface)}",end=";  ")
        paired_testdata = pair_data.load_pair_test_dataset(os.path.join("..","assets", "input", dataset_dir),participant, surface, support_size,support_include_all_conditions)
        test_loader = DataLoader(paired_testdata, batch_size=1, shuffle=False)
        acc=eval(net, test_loader,support_size, get_save_dir(participant,surface), plot=True)
        x.append(surface)
        y.append(acc)
        eval_num+=len(paired_testdata)
    title = "Accuracy (avg = " + str(round(sum(y)/len(y) * 100, 3)) + "%)"

    Utils.plot_bar(x, y, title, os.path.join(get_save_root(),participant, f"triplet network(Support Size = {support_size})_{round((time.time()-start)/eval_num *1000)}ms_per_item.png"),figsize=(5,4))
    return y

#evalute for mutilpe user
dataset_dir = "support"



if __name__ == '__main__':
    model_dir = "study1_other_1dcnn"
    root = '../assets/res/'+model_dir

    surfaces=['table','lap','monitor']

    #sort by participant
    participants = os.listdir("../assets/input/" + dataset_dir)
    participants.sort()
    embedding_size = 64
    y = []
    x = []
    for participant in participants:
        print("1")
        x.append(participant.split("_")[0])
        offset = 0
        model_path = os.path.join(root, "overall", "bestmodel.pt")
        config.start_index = 100 - (int)(embedding_size / 2) + offset
        config.embedding_size = embedding_size
        y.append(eval_traditional_network( model_path, participant))
    Utils.plot_three_bar(x, y, surfaces, os.path.join(get_save_root(), f"res.png"))



