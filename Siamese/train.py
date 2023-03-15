import os
import time

import numpy as np
from torch import nn
import Utils
import config
import network
from loss import ContrastiveLoss
import torch
from torch.utils.data import DataLoader

import pair_data
from Utils import ConfusionMatrix

OVERALL ="overall"
INPERSON = "inperson"
CROSSPERSON = "crossperson"
CROSSPERSON_20 ="crossperson_20"
CROSSPERSON_05 ="crossperson_05"
CROSSPERSON_10 ="crossperson_10"
CNN ="cnn"
RNN ="rnn"

def plot_confusion_matrix(net,data_loader,train,save,save_dir=""):
  title = "conf_train.jpg" if train else "conf_test.jpg"
  net.eval()
  class_indict = data_loader.dataset.get_label_dict()
  label = [label for _, label in class_indict.items()]
  confusion = ConfusionMatrix(num_classes=len(label), labels=label)
  with torch.no_grad():
    for target,target_label,support_set in data_loader:
        predVal = 0
        pred = -1
        for item, item_label in support_set:
            output = net(target,item)
            if output >pred:
                pred = output
                predVal = item_label
        confusion.update(predVal.numpy(),target_label.numpy())
  confusion.plot(save_dir ,title,save)
  return confusion.get_acc()


def validate(net,val_loader,test_loss,test_acc):
  # criterion = ContrastiveLoss(1.)
  criterion = nn.BCEWithLogitsLoss()
  net.eval()
  correct =0
  loss_=0
  with torch.no_grad():
    for item1,item2, labels in val_loader:
      outputs = net(item1, item2)
      loss = criterion(outputs, labels)
      loss_+=loss.item()
      predicted = outputs.data.ge(0.5)
      correct += (predicted == labels).sum().item()
  data_len = len(val_loader.dataset)
  loss_ = loss_ / len(val_loader)
  test_loss.append(loss_)
  correct = correct * 100 / data_len
  test_acc.append(correct)

# def eval(net,test_loader,save_dir="",plot=True):
#   title = "conf_test.jpg"
#   net.eval()
#   class_indict = test_loader.dataset.get_label_dict()
#   label = [label for _, label in class_indict.items()]
#   confusion = ConfusionMatrix(num_classes=len(label), labels=label)
#   with torch.no_grad():
#     for target,target_label,support_set in test_loader:
#         predVal = -1
#         # pred = 100
#         pred=-100
#         for item, item_label in support_set:
#             output = net(target,item)
#             if output > pred:
#                 pred = output
#                 predVal = item_label
#         confusion.update(predVal.numpy(),target_label.numpy())
#   if plot:
#     confusion.plot(save_dir ,title,save=True)
#   return confusion.get_acc()
#

def eval(net,test_loader,save_dir="",plot=True):
  title = "conf_test.jpg"
  net.eval()
  class_indict = test_loader.dataset.get_label_dict()
  label = [label for _, label in class_indict.items()]
  confusion = ConfusionMatrix(num_classes=len(label), labels=label)
  with torch.no_grad():
    for target,target_label,support_set in test_loader:
        labels =[]
        scores=[]
        for item, item_label in support_set:
            labels.append(item_label.item())
            scores.append(net(target,item).item())
        k=10
        idx = np.argpartition(np.array(scores),k)[:k]
        min_k_labels= np.array(labels)[idx]
        a = np.argmax(np.bincount(min_k_labels))
        a = np.array([a])
        confusion.update(a,target_label.numpy())
  if plot:
    confusion.plot(save_dir ,title,save=True)
  return confusion.get_acc()

# def eval(net,test_loader,save_dir="",plot=True):
#   title = "conf_test.jpg"
#   net.eval()
#   class_indict = test_loader.dataset.get_label_dict()
#   label = [label for _, label in class_indict.items()]
#   confusion = ConfusionMatrix(num_classes=len(label), labels=label)
#   with torch.no_grad():
#     for target,target_label,support_set in test_loader:
#         labels =[]
#         scores=[0 for i in range(8)]
#         for item, item_label in support_set:
#             scores[item_label.item()]+=net(target,item).item()
#         a = np.argmin(scores)
#         a = np.array([a])
#         confusion.update(a,target_label.numpy())
#   if plot:
#     confusion.plot(save_dir ,title,save=True)
#   return confusion.get_acc()

def train_one_epoch(net,train_loader,train_loss,train_acc):
  net.train()
  # criterion = ContrastiveLoss(1.)
  criterion = nn.BCEWithLogitsLoss()
  correct = 0
  loss_=0
  optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
  for item1,item2, labels in train_loader:

    optimizer.zero_grad()
    outputs = net(item1,item2)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    loss_+=loss.item()
    # _, predicted = torch.max(outputs.data, 1)
    predicted = outputs.data.ge(0.5)
    correct += (predicted == labels).sum().item()

  data_len = len(train_loader.dataset)
  loss_ = loss_ / len(train_loader)
  train_loss.append(loss_)
  correct =correct *100 / data_len
  train_acc.append(correct)

def get_save_root():
    return  os.path.join("..","assets","res",  NET+"_siamese_"+dataset +"_ignored_"+str(N_epoch)+"epochs_1d")
    # return os.path.join("assets", "res", 'siameseten_data__30epochs')

def get_save_dir(mode,participant=None):
  root =get_save_root()
  if mode == OVERALL:
      res = os.path.join(root,mode)
  else:
      res = os.path.join(root, mode, participant)

  if not os.path.exists(res):
    os.makedirs(res)
  return res


def train(root, mode, participant=None):
    start = time.time()
    train_dataset,val_dataset,test_dataset = pair_data.load_dataset(root,mode,participant,NET)
    save_dir = get_save_dir(mode, participant)
    print()
    print(f"Mode = {mode}, participant = {'None' if not participant else participant}")
    print("Train dataset {} , Val Dataset {}, Total {}  ,{} gestures".format(len(train_dataset), len(val_dataset), len(train_dataset) + len(val_dataset),len(train_dataset.labels)))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


    if NET ==CNN:
        # net = network.SiameseCNN(network.CNNEmbeddingNet())
        net = network.SiameseCNN(network.TAPID_CNNEmbedding())
    elif NET ==RNN:
        net = network.Siamese_RNN(len(train_loader.dataset.labels))

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    # plot_confusion_matrix(train=True,save=False)
    # plot_confusion_matrix(train=False,save=False)
    for epoch in range(N_epoch):
        train_one_epoch(net,train_loader,train_loss,train_acc)
        validate(net, test_loader, test_loss, test_acc)
        # if epoch% 25 ==0:
        print("epoch {:4} Train Loss: {:20.4f} ACC: {:20.2f}%  Test Loss: {:20.4f} ACC: {:20.2f}%"
              .format(epoch, train_loss[-1], train_acc[-1], test_loss[-1], test_acc[-1]))
        # plot_confusion_matrix(train=True,save=False)


    net.eval()
    model_path = os.path.join(save_dir,"model.pt")
    torch.save(net, model_path)

    # scripted_module = torch.jit.script(net)
    # optimize_for_mobile(scripted_module)._save_for_lite_interpreter(model_path + ".ptl")
    #
    # plot_confusion_matrix(net,train_loader,train=True, save=True, save_dir=save_dir)
    # acc = plot_confusion_matrix(net,test_loader,train=False, save=True, save_dir=save_dir)
    Utils.plot_loss(save_dir, train_loss, train_acc, test_loss, test_acc)
    acc =0
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




def eval_and_plot(mode):
    x=[f"P{i}" for i in range(1,11)]
    y=[]

    for participant in participants:
        print(f"eval participant {participant}")
        train_dataset, val_dataset, test_dataset = pair_data.load_dataset(root, mode, participant,NET)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        net = torch.load(os.path.join(get_save_root(), CROSSPERSON_05, participant, "model.pt"))
        metric =eval(net, test_loader, get_save_dir(mode,participant))
        y.append(metric)

    x.insert(0,"Average")
    y.insert(0,sum(y)/len(y))
    title = "Accuracy (avg = " + str(round(y[0] * 100, 3)) + "%)"
    Utils.plot_bar(x,y,title,os.path.join(get_save_root(),f"{mode}.png"))

def eval_onece(mode,participant):
    train_dataset, val_dataset, test_dataset = pair_data.load_dataset(root, mode, participant, NET)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    net = torch.load(os.path.join(get_save_root(), CROSSPERSON_05, participant, "model.pt"))
    metric = eval(net, test_loader, get_save_dir(mode, participant))
dataset = "ten_data_"
root = os.path.join("..","assets","input",dataset)
# participants = ['zhouyu','quyuqi','cxy','yangjingbo','zhangdan','baishuhan','yuantong','zhuqiuchen','cqs','ywn']
participants = ['zhouyu',]

N_epoch =30
NET =CNN

# #
# for participant in participants:
#     train(root, CROSSPERSON_05, participant)
eval_and_plot(CROSSPERSON_05)
# eval_and_plot(CROSSPERSON_10)
# eval_and_plot(CROSSPERSON_20)
