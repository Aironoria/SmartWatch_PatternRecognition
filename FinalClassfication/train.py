import TestData
import TripletLoss.network as triplet_net
import torch

import Utils
import network as classification_net
import os

from Utils import ConfusionMatrix


def plot_confusion_matrix(net,data_loader,train,save,save_dir="",prefix=""):
  title = "conf_train.jpg" if train else "conf_test.jpg"
  title = prefix+title
  class_indict = data_loader.dataset.get_label_dict()
  label = [label for _, label in class_indict.items()]
  confusion = ConfusionMatrix(num_classes=len(label), labels=label)
  with torch.no_grad():
    for data, labels in data_loader:
      outputs = net(data)
      _, predicted = torch.max(outputs.data, 1)
      confusion.update(predicted.cpu().numpy(), labels.cpu().numpy())
  confusion.plot(save_dir ,title,save)
  return confusion.get_acc()


def train_one_epoch(net,train_loader,acc_list,loss_list):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_func = torch.nn.CrossEntropyLoss()
    loss_=0
    correct = 0
    for step, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        output = net(x)
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()
        loss_ += loss.item()
        _, predicted = torch.max(output.data, 1)
        correct += predicted.eq(y.data.view_as(predicted)).sum().item()

    data_len = len(train_loader.dataset)
    loss_ = loss_ / len(train_loader)
    correct = correct * 100 / data_len
    acc_list.append(correct)
    loss_list.append(loss_)

def train(support_size):
    save_root = os.path.join("cjy","support_size_"+str(support_size))
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    triplet_net_path ="../assets/res/final_result/triplet/model.pt"
    train_dataset, test_dataset = TestData.load_support_and_query_dataset(root, surface="base", length=support_size,include_all_conditions=True)
    # train_dataset =test_dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    embedding_net = triplet_net.TAPID_CNNEmbedding()
    embedding_net.load_state_dict(torch.load(triplet_net_path))

    net = classification_net.Classification_Net(len(train_dataset.get_label_dict()),embedding_net)
    acc_list = []
    loss_list = []
    for i in range(100):
        print(f"epoch {i}")
        train_one_epoch(net,train_loader,acc_list, loss_list)
    # Utils.plot_loss(save_root, loss_list, acc_list, [],[])
    plot_confusion_matrix(net, train_loader, True, True, save_root)

    for surface in os.listdir(root):
        train_dataset, test_dataset = TestData.load_support_and_query_dataset(root, surface=surface, length=support_size,include_all_conditions=True)
        # test_dataset = train_dataset
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
        plot_confusion_matrix(net,test_loader,False,True,save_root,surface+"_")

dataset = "cjy"
root = os.path.join("..","assets","input",dataset)

# for i in range(1,6):
train(5)
