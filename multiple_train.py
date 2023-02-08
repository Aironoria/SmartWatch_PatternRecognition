import os

import torch.nn.functional as F
from torch.utils.mobile_optimizer import optimize_for_mobile

import Utils
import cnn
import data
import torch
from torch.utils.data import DataLoader
from Utils import ConfusionMatrix

OVERALL ="overall"
INPERSON = "inperson"
CROSSPERSON = "crossperson"
CROSSPERSON_20 ="crossperson_20"

def plot_confusion_matrix(net,data_loader,train,save,save_dir=""):
  title = "conf_train.jpg" if train else "conf_test.jpg"
  net.eval()
  class_indict = data_loader.dataset.get_label_dict()
  label = [label for _, label in class_indict.items()]
  confusion = ConfusionMatrix(num_classes=len(label), labels=label)
  with torch.no_grad():
    for data, labels in data_loader:
      outputs = net(data)
      _, predicted = torch.max(outputs.data, 1)
      confusion.update(predicted.numpy(), labels.numpy())
  confusion.plot(save_dir ,title,save)
  return confusion.get_acc()

def eval(net,test_loader,test_loss,test_acc):
  net.eval()
  correct =0
  loss_=0
  with torch.no_grad():
    for data, labels in test_loader:
      outputs = net(data)
      loss = F.cross_entropy(outputs, labels)
      loss_+=loss.item()
      _, predicted = torch.max(outputs.data, 1)
      correct += predicted.eq(labels.data.view_as(predicted)).sum().item()
  data_len = len(test_loader.dataset)
  loss_ = loss_ / len(test_loader)
  test_loss.append(loss_)
  correct = correct * 100 / data_len
  test_acc.append(correct)


def train_one_epoch(net,train_loader,train_loss,train_acc):
  net.train()
  correct = 0
  loss_=0
  optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
  for batch_idx, (data, labels) in enumerate(train_loader):

    optimizer.zero_grad()
    outputs = net(data)

    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    optimizer.step()

    loss_+=loss.item()
    _, predicted = torch.max(outputs.data, 1)
    correct += predicted.eq(labels.data.view_as(predicted)).sum().item()
    # if batch_idx % 2 == 0:
    #   print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tAcc:{:.6f}'.format(
    #     epoch, batch_idx * len(data), len(train_loader.dataset),
    #     100. * batch_idx / len(train_loader), loss.jpg.item(),correct/(log_interval * len(data))),end="\n")
  data_len = len(train_loader.dataset)
  loss_ = loss_ / len(train_loader)
  train_loss.append(loss_)
  correct =correct *100 / data_len
  train_acc.append(correct)


def get_save_root():
    return  os.path.join("assets","res",  dataset +"_"+str(N_epoch)+"epochs")

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
    train_dataset,test_dataset = data.load_dataset(root,mode,participant)
    save_dir = get_save_dir(mode, participant)
    print()
    print(f"Mode = {mode}, participant = {'None' if not participant else participant}")
    print("Train dataset {} , Test Dataset {}, Total {} ".format(len(train_dataset), len(test_dataset), len(train_dataset) + len(test_dataset)))
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # net = cnn.Net(len(train_loader.dataset.labels))

    net = cnn.RNN(len(train_loader.dataset.labels))
    # net = torch.load(root +".pt")
    # net = torch.load("assets/res/11-15_len(49)_with10-27_sampled1_30epochs_28378/11-15_len(49)_with10-27_sampled1.pt")


    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    # plot_confusion_matrix(train=True,save=False)
    # plot_confusion_matrix(train=False,save=False)
    for epoch in range(N_epoch):
        train_one_epoch(net,train_loader,train_loss,train_acc)
        eval(net, test_loader, test_loss, test_acc)
        # if epoch% 25 ==0:
        print("epoch {:4} Train Loss: {:20.4f} ACC: {:20.2f}%  Test Loss: {:20.4f} ACC: {:20.2f}%"
              .format(epoch, train_loss[-1], train_acc[-1], test_loss[-1], test_acc[-1]))
        # plot_confusion_matrix(train=True,save=False)


    net.eval()
    model_path = os.path.join(save_dir,"model.pt")
    torch.save(net, model_path)

    scripted_module = torch.jit.script(net)
    optimize_for_mobile(scripted_module)._save_for_lite_interpreter(model_path + ".ptl")

    plot_confusion_matrix(net,train_loader,train=True, save=True, save_dir=save_dir)
    acc = plot_confusion_matrix(net,test_loader,train=False, save=True, save_dir=save_dir)
    Utils.plot_loss(save_dir, train_loss, train_acc, test_loss, test_acc)
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
    Utils.plot_bar(x,y,get_save_root(),f"{mode}.png")

dataset = "ten_data"
root = os.path.join("assets","input",dataset)
N_epoch =100


train_and_plot(INPERSON)
train_and_plot(CROSSPERSON)
train_and_plot(CROSSPERSON_20)
