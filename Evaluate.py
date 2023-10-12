#evaluating traditional model and triplet loss based model on new participant's data

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

from TripletLoss import network

device = torch.device('cpu')


def plot_confusion_matrix(net,data_loader,save_dir=""):
  title = "conf_test_cnn.jpg"
  net.eval()
  net.to(device)
  class_indict = data_loader.dataset.get_label_dict()
  label = [label for _, label in class_indict.items()]
  confusion = ConfusionMatrix(num_classes=len(label), labels=label)
  with torch.no_grad():
    for data, labels in data_loader:
      data = data.to(device)
      labels = labels.to(device)
      outputs = net(data)
      _, predicted = torch.max(outputs.data, 1)
      confusion.update(predicted.cpu().numpy(), labels.cpu().numpy())
  confusion.plot(save_dir ,title, save=True)
  return confusion.get_acc()

def calc_distance(x1,x2):
    # return (x1-x2).pow(2).sum(1)
    dist = nn.PairwiseDistance(p=2)
    return dist(x1,x2)


def eval(net,test_loader,support_size,save_dir="",plot=True):
  knn_n = 1
  title = f"conf_test_triplet_{support_size}.png"
  net.eval()
  class_indict = test_loader.dataset.get_label_dict()
  label = [label for _, label in class_indict.items()]
  print("confusion matrix", label)
  with torch.no_grad():
      support = pd.DataFrame( [ (net(i[0].unsqueeze(0))[0].numpy(), i[1].item() )for i in test_loader.dataset[0][2]],columns=["embedding","label"])


  confusion = ConfusionMatrix(num_classes=len(label), labels=label)
  with torch.no_grad():
    for target,target_label,support_set in test_loader:
        embedding =net(target)[0]
        scores = [ calc_distance(embedding,torch.from_numpy(i)) for i in support["embedding"]]


        scores = torch.stack(scores).squeeze()

        label = Utils.weighted_knn(scores,support["label"].values,knn_n)
        predVal = torch.tensor(label).unsqueeze(0)

        confusion.update(predVal.numpy(),target_label.numpy())
  if plot:
    confusion.plot(save_dir ,title,save=True)
  return confusion.get_acc()



def get_save_root():
    # return os.path.join("assets", "res", "cnn_" + dataset + "_ignored_3gestures_" + str(N_epoch) + "1d")
    return  os.path.join("assets","res",  "study1_use_triplet_"+dataset_dir)


def get_save_dir(surface):
  root =get_save_root()
  res = os.path.join(root,surface)
  if not os.path.exists(res):
    os.makedirs(res)
  return res






dataset_dir = "cjy"

def eval_traditional_network():

    net_dir = "assets/res/final_result/cnn/bestmodel.pt"
    net = torch.load(net_dir)
    x =[]
    y= []
    eval_num=0
    start = time.time()
    print("evaluating traditional cnn")
    for surface in os.listdir(os.path.join("assets", "input", dataset_dir)):
        if surface =="new":
            continue
        dataset = data.load_test_dataset(os.path.join("assets", "input", dataset_dir),surface)
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        acc=plot_confusion_matrix(net, test_loader, get_save_dir(surface))
        x.append(surface)
        y.append(acc)
        eval_num+=len(dataset)
    title = "Accuracy (avg = " + str(round(sum(y)/len(y) * 100, 3)) + "%)"
    Utils.plot_bar(x,y,title,os.path.join(get_save_root(),f"traditional cnn_{round((time.time()-start)/eval_num *1000)}ms_per_item.png"))

def eval_triplet_network(support_size,support_include_all_conditions=False):
    net = network.TAPID_CNNEmbedding()
    net.load_state_dict(torch.load("assets/res/study1_use_triplet/overall/model.pt"))
    x = []
    y = []
    eval_num= 0
    start = time.time()
    print("evaluating triplet network, support size = " + str(support_size))
    surfaces = os.listdir(os.path.join("assets", "input", dataset_dir))
    surfaces = ['base','lap','wall','left','new']
    for surface in surfaces:
    # for surface in ["new"]:
        print("eval surface: " + surface,end=";  ")
        paired_testdata = pair_data.load_pair_test_dataset(os.path.join("assets", "input", dataset_dir), surface, support_size,support_include_all_conditions)
        test_loader = DataLoader(paired_testdata, batch_size=1, shuffle=False)
        acc=eval(net, test_loader,support_size, get_save_dir(surface), plot=True)
        x.append(surface)
        y.append(acc)
        eval_num+=len(paired_testdata)
    title = "Accuracy (avg = " + str(round(sum(y)/len(y) * 100, 3)) + "%)"

    Utils.plot_bar(x, y, title, os.path.join(get_save_root(), f"triplet network(Support Size = {support_size})_{round((time.time()-start)/eval_num *1000)}ms_per_item.png"),figsize=(5,4))


# config.ignored_label = ['make_fist','touchdown','touchup','nothing']
# eval_traditional_network()
margin = 0.01
# for i in range(1,6):
#     eval_triplet_network(i, True)
eval_triplet_network(1,True)
eval_triplet_network(5,True)
