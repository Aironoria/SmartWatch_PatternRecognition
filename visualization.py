from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

import os
import time
import numpy as np
import Utils

import torch
from torch.utils.data import DataLoader

import cnn
import pair_data
from TripletLoss import network

def get_save_dir():
    dir = os.path.join("assets", "visualization", dataset_dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir
def tsne_plot(x, y, labels, save_dir="",title=""):

    # Create a two dimensional t-SNE projection of the embeddings
    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(x)
    # Plot those points as a scatter plot and label them based on the pred labels
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8, 8))
    num_categories = 7
    for label in range(len(labels)):
        indices =( y == label).squeeze()
        ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], c=np.array(cmap(label)).reshape(1, 4), label=labels[label],
                   alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.savefig(os.path.join(save_dir,title+".png"))



def visualize(data_loader,embedding_net="triplet",method="tsne",title=""):
    if embedding_net == "triplet":
        net = network.TAPID_CNNEmbedding()
        net.load_state_dict(torch.load("assets/res/final_result/triplet/model.pt"))
    elif embedding_net == "cnn":
        net = cnn.oneDCNN()
        net.load_state_dict(torch.load("assets/res/study1_final/overall/bestmodel.pt"))
    else:
        raise ValueError("net must be either triplet or cnn")
    net.eval()
    class_indict = data_loader.dataset.get_label_dict()
    labels = [label for _, label in class_indict.items()]
    title = title+"_" +embedding_net+"_"+method
    with torch.no_grad():
        for target, target_label, support_set in test_loader:
            embedding,_ = net(target)
            # embedding = target.reshape(len(target_label),-1)
            if method == "tsne":
                tsne_plot(embedding.numpy(),target_label.unsqueeze(-1).numpy(),labels,get_save_dir(),title)




if __name__ == '__main__':
    dataset_dir = "cjy"
    net_dir = "assets/res/final_result/triplet/model.pt"
    net = network.TAPID_CNNEmbedding()
    for surface in os.listdir(os.path.join("assets", "input", dataset_dir)):
    # for surface in ["base"]:
        net.load_state_dict(torch.load(net_dir))
        paired_testdata = pair_data.load_pair_test_dataset(os.path.join("assets", "input", "cjy"), surface,
                                                           1, True)
        test_loader = DataLoader(paired_testdata, batch_size=len(paired_testdata), shuffle=False)
        visualize(test_loader,embedding_net="cnn",title=surface)