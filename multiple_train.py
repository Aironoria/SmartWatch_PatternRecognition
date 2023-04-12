import os
import time

import torch.nn.functional as F
from torch.utils.mobile_optimizer import optimize_for_mobile

import Utils
import cnn
import config
import data
import torch
from torch.utils.data import DataLoader
from Utils import ConfusionMatrix

OVERALL ="overall"
INPERSON = "inperson"
CROSSPERSON = "crossperson"
CROSSPERSON_20 ="crossperson_20"
CROSSPERSON_05 ="crossperson_05"
CROSSPERSON_10 ="crossperson_10"

# if torch.cuda.is_available():
#     device = torch.device('cuda')
# elif torch.backends.mps.is_available():
#     device = torch.device('mps')
# else:
#     device = torch.device('cpu')
device = torch.device('cpu')
def plot_confusion_matrix(net,data_loader,train,save,save_dir="",prefix=""):
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
      outputs = net(data)
      _, predicted = torch.max(outputs.data, 1)
      confusion.update(predicted.cpu().numpy(), labels.cpu().numpy())
  confusion.plot(save_dir ,title,save)
  return confusion.get_acc()

def eval(net,test_loader,test_loss,test_acc):
  net.eval()
  correct =0
  loss_=0
  net.to(device)
  with torch.no_grad():
    for data, labels in test_loader:
      data = data.to(device)
      labels = labels.to(device)
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
  return correct


def train_one_epoch(net,train_loader,train_loss,train_acc):
  net.train()
  correct = 0
  loss_=0
  optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
  net.to(device)
  for batch_idx, (data, labels) in enumerate(train_loader):
    data = data.to(device)
    labels = labels.to(device)
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
    # return os.path.join("assets", "res", "cnn_" + dataset + "_ignored_3gestures_" + str(N_epoch) + "1d")
    return  os.path.join("assets","res",  "final_result1")


def get_save_dir(mode,participant=None):
  root =get_save_root()

  if mode == OVERALL:
      res = os.path.join(root,mode)
  else:
      res = os.path.join(root, mode, participant)

  if not os.path.exists(res):
    os.makedirs(res)
  return res


def train(root, mode, participant=None,n=None):
    start = time.time()
    train_dataset,test_dataset = data.load_dataset(root,mode,participant,n)
    if not n == None:
        mode = mode + "_" + str(n)
    save_dir = get_save_dir(mode, participant)
    print()
    print(f"Mode = {mode}, participant = {'None' if not participant else participant}")
    print("Train dataset {} , Test Dataset {}, Total {}, {} gestures ".format(len(train_dataset), len(test_dataset), len(train_dataset) + len(test_dataset),len(train_dataset.labels)))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # net = cnn.Net(len(train_loader.dataset.labels))
    if Net == 'cnn':
        # net = cnn.Net(len(train_loader.dataset.labels))
        net = cnn.oneDCNN()
    elif Net =='rnn':
        net = cnn.RNN(len(train_loader.dataset.labels))
    # net = torch.load(root +".pt")
    # net = torch.load("assets/res/11-15_len(49)_with10-27_sampled1_30epochs_28378/11-15_len(49)_with10-27_sampled1.pt")


    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    # plot_confusion_matrix(train=True,save=False)
    # plot_confusion_matrix(train=False,save=False)
    bestscore, bestepoch=0,0
    model_path = os.path.join(save_dir,"bestmodel.pt")
    for epoch in range(N_epoch):
        train_one_epoch(net,train_loader,train_loss,train_acc)
        eval(net, test_loader, test_loss, test_acc)


        # if epoch% 25 ==0:
        print("epoch {:4} Train Loss: {:20.4f} ACC: {:20.2f}%  Test Loss: {:20.4f} ACC: {:20.2f}%"
              .format(epoch, train_loss[-1], train_acc[-1], test_loss[-1], test_acc[-1]))
        if (test_acc[-1] > bestscore):
            bestscore = test_acc[-1]
            bestepoch = epoch
            torch.save(net, model_path)
            print("model saved.")
        # plot_confusion_matrix(train=True,save=False)

    print(f"best epoch: {bestepoch}, best acc{bestscore}")


    # model_path = os.path.join(save_dir, "lastmodel.pt")
    # torch.save(net, model_path)
    # plot_confusion_matrix(net, train_loader, train=True, save=True, save_dir=save_dir,prefix="last")
    # acc = plot_confusion_matrix(net, test_loader, train=False, save=True, save_dir=save_dir,prefix="last")

    model_path = os.path.join(save_dir, "bestmodel.pt")
    net = torch.load(model_path)
    plot_confusion_matrix(net, train_loader, train=True, save=True, save_dir=save_dir, prefix="best")
    acc = plot_confusion_matrix(net, test_loader, train=False, save=True, save_dir=save_dir, prefix="best")

    Utils.plot_loss(save_dir, train_loss, train_acc, test_loss, test_acc)
    print(time.time() - start)
    return acc


def train_and_plot(mode,n=None):
    x=[f"P{i}" for i in range(1,11)]
    y=[]

    for participant in participants:
        metric = train(root,mode,participant,n)
        y.append(metric)

    # if mode ==INPERSON:
    #     x.insert(0,"Overall")
    #     y.insert(0,train(root,OVERALL,n))
    # else:
    x.insert(0,"Average")
    y.insert(0,sum(y)/len(y))
    title = "Accuracy (avg = " + str(round(y[0]*100,3)) + "%)"
    Utils.plot_bar(x,y,title,os.path.join(get_save_root(),f"{mode+('' if n ==None else '_'+str(n))}.png"))


def eval_and_plot(mode):
    x=[f"P{i}" for i in range(1,11)]
    y=[]

    for participant in participants:
        print(f"eval participant {participant}")
        train_dataset, test_dataset = data.load_dataset(root, mode, participant)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        net = torch.load(os.path.join(get_save_root(), mode, participant, "bestmodel.pt"))
        metric =eval(net, test_loader,[],[])/100
        y.append(metric)

    x.insert(0,"Average")
    y.insert(0,sum(y)/len(y))
    title = "Accuracy (avg = " + str(round(y[0] * 100, 3)) + "%)"
    Utils.plot_bar(x,y,title,os.path.join(get_save_root(),f"{mode}.png"))
    # Utils.plot_bar(x,y,title,'result.png')

dataset = "ten_data_"
root = os.path.join("assets","input",dataset)
participants = ['zhouyu','quyuqi','cxy','yangjingbo','zhangdan','baishuhan','yuantong','zhuqiuchen','cqs','ywn']
N_epoch =81
# config.ignored_label = ['touchdown','touchup']
Net = config.network
train(root,OVERALL)
# train_and_plot(INPERSON)
# train_and_plot(CROSSPERSON)
# train_and_plot(CROSSPERSON_05)
# train_and_plot(CROSSPERSON_10)
# train_and_plot(CROSSPERSON_20)
# train(root,OVERALL)
# for n in range(5,101,5):
#     train_and_plot(INPERSON, n)
#
# eval_and_plot(CROSSPERSON)
# eval_and_plot(CROSSPERSON_05)
# eval_and_plot(CROSSPERSON_10)
# eval_and_plot(CROSSPERSON_20)