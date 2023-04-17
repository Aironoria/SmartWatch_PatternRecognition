import os
import time
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
  title = f"conf_test_triplet_{support_size}.png"
  net.eval()
  class_indict = test_loader.dataset.get_label_dict()
  label = [label for _, label in class_indict.items()]
  print("confusion matrix", label)
  confusion = ConfusionMatrix(num_classes=len(label), labels=label)
  with torch.no_grad():
    for target,target_label,support_set in test_loader:
        predVal = -1
        pred = 1000
        # pred=-100
        for item, item_label in support_set:
            output = calc_distance(net(target), net(item))
            if output < pred:
                pred = output
                predVal = item_label
        confusion.update(predVal.numpy(),target_label.numpy())
  if plot:
    confusion.plot(save_dir ,title,save=True)
  return confusion.get_acc()


# def eval(net,test_loader,support_size,save_dir="",plot=True):
#   title = f"conf_test_triplet_{support_size}.png"
#   net.eval()
#   class_indict = test_loader.dataset.get_label_dict()
#   label = [label for _, label in class_indict.items()]
#   print(label)
#   confusion = ConfusionMatrix(num_classes=len(label), labels=label)
#   with torch.no_grad():
#       for target, target_label, support_set in test_loader:
#           labels =[]
#           scores=[]
#           for item, item_label in support_set:
#               labels.append(item_label.item())
#               scores.append(calc_distance(net(target), net(item)).item())
#           k=5
#           idx = np.argpartition(np.array(scores),k)[:k]
#           min_k_labels= np.array(labels)[idx]
#           a = np.argmax(np.bincount(min_k_labels))
#           a = np.array([a])
#           confusion.update(a,target_label.numpy())
#   if plot:
#     confusion.plot(save_dir ,title,save=True)
#   return confusion.get_acc()


def get_save_root():
    # return os.path.join("assets", "res", "cnn_" + dataset + "_ignored_3gestures_" + str(N_epoch) + "1d")
    return  os.path.join("assets","res",  dataset_dir+"_include_all_conditions")


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
    net_dir = "assets/res/final_result/triplet/model.pt"
    net = network.TAPID_CNNEmbedding()
    x = []
    y = []
    eval_num= 0
    start = time.time()
    print("evaluating triplet network, support size = " + str(support_size))
    for surface in os.listdir(os.path.join("assets", "input", dataset_dir)):
    # for surface in ["new"]:

        print("eval surface: " + surface,end=";  ")
        net.load_state_dict(torch.load(net_dir))
        paired_testdata = pair_data.load_pair_test_dataset(os.path.join("assets", "input", "cjy"), surface, support_size,support_include_all_conditions)
        test_loader = DataLoader(paired_testdata, batch_size=1, shuffle=False)
        acc=eval(net, test_loader,support_size, get_save_dir(surface), plot=True)
        x.append(surface)
        y.append(acc)
        eval_num+=len(paired_testdata)
    title = "Accuracy (avg = " + str(round(sum(y)/len(y) * 100, 3)) + "%)"

    Utils.plot_bar(x, y, title, os.path.join(get_save_root(), f"triplet network(Support Size = {support_size})_{round((time.time()-start)/eval_num *1000)}ms_per_item.png"))


# config.ignored_label = ['make_fist','touchdown','touchup','nothing']
# eval_traditional_network()
for i in range (1,6):
    eval_triplet_network(i,True)