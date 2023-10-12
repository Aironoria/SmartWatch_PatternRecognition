import os
import time

import numpy as np
from torch import nn
import pandas as pd
import Utils
import config
import network
from loss import TripletLoss
from loss import calc_distance
import torch
from torch.utils.data import DataLoader
import  torch.nn.functional as F
import triplet_data
from Utils import ConfusionMatrix
import pair_data

OVERALL ="overall"
INPERSON = "inperson"
CROSSPERSON = "crossperson"
CROSSPERSON_20 ="crossperson_20"
CROSSPERSON_05 ="crossperson_05"
CROSSPERSON_10 ="crossperson_10"
CNN ="cnn"
RNN ="rnn"

def validate(net,val_loader,test_loss):
  criterion = TripletLoss()
  # criterion = nn.BCEWithLogitsLoss()
  net.eval()
  correct =0
  loss_=0
  with torch.no_grad():
    for anchor,positive,negative in val_loader:
      loss = criterion(net(anchor), net(positive),net(negative))
      loss_+=loss.item()

  data_len = len(val_loader.dataset)
  loss_ = loss_ / len(val_loader)
  test_loss.append(loss_)


def eval(net, test_loader, support_size, save_dir="", plot=True):
    knn_n = 1
    title = f"conf_test_triplet_{support_size}.png"
    net.eval()
    class_indict = test_loader.dataset.get_label_dict()
    label = [label for _, label in class_indict.items()]
    print("confusion matrix", label)
    with torch.no_grad():
        support = pd.DataFrame([(net(i[0].unsqueeze(0))[0].numpy(), i[1].item()) for i in test_loader.dataset[0][2]],
                               columns=["embedding", "label"])

    confusion = ConfusionMatrix(num_classes=len(label), labels=label)
    with torch.no_grad():
        for target, target_label, support_set in test_loader:
            embedding = net(target)[0]
            scores = [calc_distance(embedding, torch.from_numpy(i)) for i in support["embedding"]]

            scores = torch.stack(scores).squeeze()

            label = Utils.weighted_knn(scores, support["label"].values, knn_n)
            predVal = torch.tensor(label).unsqueeze(0)

            confusion.update(predVal.numpy(), target_label.numpy())
    if plot:
        confusion.plot(save_dir, title, save=True)
    return confusion.get_acc()


def train_one_epoch(net,train_loader,train_loss,margin):
  net.train()
  criterion = TripletLoss(margin)
  correct = 0
  loss_=0
  optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
  for anchor,positive,negative in train_loader:

    optimizer.zero_grad()
    loss = criterion(net(anchor),net(positive),net(negative))
    loss.backward()
    optimizer.step()

    loss_+=loss.item()
    # _, predicted = torch.max(outputs.data, 1)

  data_len = len(train_loader.dataset)
  loss_ = loss_ / len(train_loader)
  train_loss.append(loss_)

def get_save_root():
    # return  os.path.join("..","assets","res",  NET+"_triplet_"+dataset +"_ignored_"+str(N_epoch)+"epochs_1d")
    return os.path.join("..","assets", "res", 'study1_use_triplet')

def get_save_dir(mode,participant=None,n=None):
  root =get_save_root()
  if mode == OVERALL:
      res = os.path.join(root,mode)
  else:
      mode = mode + ("_" + str(n).zfill(2) if  n != None else "")
      res = os.path.join(root, mode, participant)

  if not os.path.exists(res):
    os.makedirs(res)
  return res


def train(root, mode, participant=None,margin=1,n=None):
    start = time.time()
    train_dataset,val_dataset,test_dataset = triplet_data.load_dataset(root,mode,participant,NET,n)
    save_dir = get_save_dir(mode, participant,n)
    print()
    print(f"Mode = {mode + ('' if n == None else '_' + str(n))}, participant = {'None' if not participant else participant}")
    print("Train dataset {} , Val Dataset {}, Total {}  ,{} gestures".format(len(train_dataset), len(val_dataset), len(train_dataset) + len(val_dataset),len(train_dataset.labels)))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if NET ==CNN:
        net = network.TAPID_CNNEmbedding()
    elif NET ==RNN:
        net = network.Siamese_RNN(len(train_loader.dataset.labels))

    train_loss = []
    test_loss = []
    for epoch in range(N_epoch):
        train_one_epoch(net,train_loader,train_loss,margin=margin)
        validate(net, val_loader, test_loss)
        # if epoch% 25 ==0:
        print("epoch {:4} Train Loss: {:20.4f}  Test Loss: {:20.4f} "
              .format(epoch, train_loss[-1], test_loss[-1]))
        # plot_confusion_matrix(train=True,save=False)
    net.eval()

    model_path = os.path.join(save_dir,"model.pt")
    torch.save(net.state_dict(), model_path)

    Utils.plot_loss(save_dir, train_loss, [], test_loss, [])

    acc = eval(net, test_loader, 5, save_dir,plot=True)

    print(time.time() - start)
    return acc


def train_and_plot(mode):
    x=[f"P{i}" for i in range(1,11)]
    y=[]

    for participant in os.listdir(root):
        metric = train(root,mode,participant)
        y.append(metric)

    if mode ==INPERSON:
        x.insert(0,"Overall")
        y.insert(0,train(root,OVERALL))
    else:
        x.insert(0,"Average")
        y.insert(0,sum(y)/len(y))
    Utils.plot_bar(x,y,os.path.join(get_save_root(),f"{mode}.png"))







dataset = "ten_data_"
root = os.path.join("..","assets","input",dataset)
participants = ['zhouyu','quyuqi','cxy','yangjingbo','zhangdan','baishuhan','yuantong','zhuqiuchen','cqs','ywn']

N_epoch = 100
NET =CNN

def train_test_plot(mode ,n=None):
    x=[f"P{i}" for i in range(len(participants))]
    y=[]
    for participant in participants:
        y.append(train(root,mode,participant,margin=0.01,n=n))
    title = "Accuracy (avg = " + str(round(y[0] * 100, 3)) + "%)"
    Utils.plot_bar(x, y, title, os.path.join(get_save_root(), f"{mode + ('' if n == None else '_' + str(n))}.png"))

# train_test_plot(INPERSON)
#
# train_test_plot(CROSSPERSON,0)
# train_test_plot(CROSSPERSON,5)
# train(root,OVERALL,margin=0.01)


