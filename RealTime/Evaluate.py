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
import matplotlib.pyplot as plt

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
    return  os.path.join("..","assets","res",  model_dir+"_"+dataset_dir)


def get_save_dir(participant,surface):
  root =get_save_root()
  res = os.path.join(root,participant,surface)
  if not os.path.exists(res):
    os.makedirs(res)
    print("create dir: " + res)

  return res






def eval_triplet_network(support_size,net_path,participant,support_include_all_conditions=False):
    net = network.TAPID_CNNEmbedding()
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

# config.ignored_label = ['make_fist','touchdown','touchup','nothing']
# eval_traditional_network()
# margin = 0.01
# # for i in range(1,6):
# #     eval_triplet_network(i, True)
# eval_triplet_network(1,True)
# eval_triplet_network(5,True)

def eval_method_with_different_augmentation():
    for jitter in [False,True]:
        for time in [False,True]:
            for magnitude in [False,True]:
                model_dir = "overall"
                if jitter:
                    model_dir += "_jitter"
                if time:
                    model_dir += "_time"
                if magnitude:
                    model_dir += "_mag"
                config.model_dir = model_dir
                model_path = os.path.join("../assets/res/study1_use_triplet_real_segmentation_30",model_dir,"model.pt")
                # eval_triplet_network(1,model_path,True)
                eval_triplet_network(5,model_path,True)


def plot_three_bar(x,y,label,save_path,xlabel,title):

    # Number of participants
    n_participants = len(x)
    # figsize
    # Creating bar plot
    # fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(10,5))

    # Setting the positions and width for the bars
    ind = np.arange(n_participants)
    width = 0.25

    y = np.array(y)
    # Plotting
    for i in range(len(label)):
        ax.bar(ind + i * width, y[:, i], width, label=label[i])
        #plt text value
        for j in range(len(x)):
            ax.text(ind[j] + i * width, y[j, i] + 0.005, str(round(y[j, i], 3)) , ha='center', va='bottom', fontsize=10)

    # Adding labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Acc')
    # ax.set_title('Bar plot for participants x1, x2, x3 with values x, y, z')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(x)
    ax.legend(loc='lower right')
    #ylim
    # plt.ylim(0.6,1)
    # Display the plot
    # plt.show()
    plt.title(title)
    plt.savefig(save_path,bbox_inches='tight')

#evalute for mutilpe user
dataset_dir = "support"


if __name__ == '__main__':
    model_dir = "study1_use_triplet_real_segmentation_embedding_conv_3"
    root = '../assets/res/'+model_dir

    surfaces=['table','lap','monitor']

    #sort by participant
    participants = os.listdir("../assets/input/" + dataset_dir)
    participants.sort()
    # participants = ["p1_lhl","p5_gbz"]
    support_sizes= [1,2,3,4,5]
    acc = []
    for support_size in support_sizes:
        y = []
        x = []
        for participant in participants:
            embedding_size = 64
            x.append(participant.split("_")[0])
            offset = 0
            model_path = os.path.join(root, str(embedding_size), "model.pt")
            config.start_index = 100 - (int)(embedding_size / 2) + offset
            config.embedding_size = embedding_size
            y.append(eval_triplet_network(support_size, model_path, participant, True))
        acc.append(np.mean(y,axis=0).round(3))
        plot_three_bar(x, y, surfaces, os.path.join(get_save_root(), f"support_size_{support_size}.png"),
                       "Participant", f"Average acc on ({surfaces[0]},{surfaces[1]},{surfaces[2]}) : {np.mean(y,axis=0).round(3)}")
    plot_three_bar(support_sizes, acc, surfaces, os.path.join(get_save_root(), f"overall.png"),"Support Size",
                   f"Average acc on ({surfaces[0]},{surfaces[1]},{surfaces[2]}) as Support Size increase")


