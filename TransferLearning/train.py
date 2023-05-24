#evaluating traditional model and triplet loss based model on new participant's data

import os
import random
import time
import copy
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.mobile_optimizer import optimize_for_mobile
import numpy as np
import Utils
import cnn
import torch
from torch.utils.data import DataLoader

import pair_data
import data
from TripletLoss import triplet_data
from Utils import ConfusionMatrix

from TripletLoss import network as triplet_network

device = torch.device('cpu')


def plot_confusion_matrix(net,data_loader,train,save,plot=True,save_dir="",prefix=""):
  title = "conf_train.jpg" if train else "conf_test.jpg"
  title = prefix+title
  net.eval()
  net.to(device)
  class_indict = data_loader.dataset.get_label_dict()
  label = [label for _, label in class_indict.items()]
  confusion = ConfusionMatrix(num_classes=len(label), labels=label)
  with torch.no_grad():
    for data, labels in data_loader:
      data = data.to(device)
      labels = labels.to(device)
      embedding,outputs = net(data)
      _, predicted = torch.max(outputs.data, 1)
      confusion.update(predicted.cpu().numpy(), labels.cpu().numpy())
  if plot:
    confusion.plot(save_dir ,title,save)
  return confusion.get_acc() , confusion.get_f1_score()


def get_save_root():
    # return os.path.join("assets", "res", "cnn_" + dataset + "_ignored_3gestures_" + str(N_epoch) + "1d")
    return  os.path.join("res",  dataset_dir)


def get_save_dir(surface):
  root =get_save_root()
  res = os.path.join(root,surface)
  if not os.path.exists(res):
    os.makedirs(res)
  return res




def fine_tune(model,support_data):
    train_loader = DataLoader(support_data, batch_size=16, shuffle=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fcout = nn.Linear(128, len(support_data.labels))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    best_loss=100
    for epoch in range(80):
        loss_=0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)[-1]

            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_+=loss.item()
        if loss_<best_loss:
            best_loss=loss_
            best_model_weights = copy.deepcopy(model.state_dict())
        if epoch % 10 == 0:
            print("epoch: " + str(epoch) + " loss: " + str(loss.item()))
    model.load_state_dict(best_model_weights)
    plot_confusion_matrix(model, train_loader, True, True, save_dir=get_save_dir("new"))
    return model


def eval_triplet_network(support_size,support_include_all_conditions=False):

    net = cnn.oneDCNN()
    # net.load_state_dict(torch.load("../assets/res/study1_final/overall/bestmodel.pt"))

    # net = triplet_network.TAPID_CNNEmbedding()
    # state1 = torch.load("../assets/res/study1_final/overall/bestmodel.pt")
    state = torch.load("../assets/res/final_result00/overall/model.pt")
    state["fcout.weight"] = torch.rand((10,128))
    state["fcout.bias"] = torch.rand(10)
    net.load_state_dict(state)
    support = data.load_support_dataset(os.path.join("..", "assets", "input", "cjy"), "base", support_size, True)
    net = fine_tune(net, support)
    x=[]
    y =[]
    eval_num=0
    start = time.time()
    random.seed(0)
    for surface in os.listdir(os.path.join("..","assets", "input", dataset_dir)):
        if surface =="new":
            continue
        print("eval surface: " + surface,end=";  ")
        query_data = data.load_query_dataset(os.path.join("..", "assets", "input", "cjy"), surface)
        test_loader = DataLoader(query_data, batch_size=1, shuffle=False)
        acc, f1 = plot_confusion_matrix(net,test_loader,False,True,save_dir=get_save_dir(surface),prefix=str(support_size)+"_")
        x.append(surface)
        y.append(acc)
        eval_num += len(query_data)
    title = "Accuracy (avg = " + str(round(sum(y)/len(y) * 100, 3)) + "%)"
    Utils.plot_bar(x, y, title, os.path.join(get_save_root(), f"triplet network(Support Size = {support_size})_{round((time.time()-start)/eval_num *1000)}ms_per_item.png"))



dataset_dir = "cjy"
for i in range (1,6):
    eval_triplet_network(i,True)
