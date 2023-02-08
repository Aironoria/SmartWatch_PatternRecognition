import os

import torch.nn.functional as F
from torch.utils.mobile_optimizer import optimize_for_mobile

import Utils
import cnn
from single import data
import torch
from torch.utils.data import DataLoader
from Utils import ConfusionMatrix


def plot_confusion_matrix(train,save,save_dir=""):
  data_loader = train_loader if train else test_loader
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

def eval(epoch):
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


def train_one_epoch(epoch):
  net.train()
  correct = 0
  loss_=0
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


def get_save_dir(epoch,dataset):
  root="assets/res"
  size = len(train_dataset) +len(test_dataset)
  res= os.path.join( root, dataset+"_"+ str(epoch) +"epochs_"+ str(size)   ) #"time_domain"

  if not os.path.exists(res):
    os.makedirs(res)
  return res


for i in [0]:
  dataset = "10-27_11-15_12-04_len65_sampled"
  # root = "assets/input/" +dataset
  root =os.path.join("assets", "input", dataset)
  train_dataset , test_dataset  = data.load_dataset(root)
  print("Train dataset {} , Test Dataset {}, Total {} ".format(len(train_dataset), len(test_dataset),len(train_dataset)+len(test_dataset)))
  train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

  # net = cnn.Net(len(train_loader.dataset.labels))

  net = cnn.RNN(len(train_loader.dataset.labels))
  # net = torch.load(root +".pt")
  # net = torch.load("assets/res/11-15_len(49)_with10-27_sampled1_30epochs_28378/11-15_len(49)_with10-27_sampled1.pt")
  optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

  train_loss = []
  train_acc = []
  test_loss = []
  test_acc = []

  # plot_confusion_matrix(train=True,save=False)
  # plot_confusion_matrix(train=False,save=False)
  N_epoch =50
  for epoch in range(N_epoch):
    train_one_epoch(epoch)
    eval(epoch)
    # if epoch% 25 ==0:
    print("epoch {:4} Train Loss: {:20.4f} ACC: {:20.2f}%  Test Loss: {:20.4f} ACC: {:20.2f}%"
            .format(epoch, train_loss[-1], train_acc[-1],test_loss[-1],test_acc[-1]))
    # plot_confusion_matrix(train=True,save=False)

  pic_save_dir = get_save_dir(N_epoch,dataset)

  net.eval()
  model_path = os.path.join(pic_save_dir , dataset)
  torch.save(net, model_path +'.pt')

  scripted_module = torch.jit.script(net)
  optimize_for_mobile(scripted_module)._save_for_lite_interpreter(model_path + ".ptl")

  plot_confusion_matrix(train=True,save=True,save_dir=pic_save_dir)
  plot_confusion_matrix(train=False, save=True,save_dir=pic_save_dir)
  Utils.plot_loss(pic_save_dir, train_loss, train_acc, test_loss, test_acc)

