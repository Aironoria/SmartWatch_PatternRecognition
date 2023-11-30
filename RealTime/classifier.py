from TripletLoss import network as triplet_network
import torch
import pandas as pd
import pair_data
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import supportDataset
class TripletClassifier:
    def __init__(self, model_path = None, support_set_path = None):
        # model_path = "../assets/res/final_result_margin_0.01/overall/model.pt"
        model_path ="../assets/res/study1_use_triplet_real_segmentation_margin/overall_margin_2/model.pt"
        # model_path = "../assets/res/study1_use_triplet_augmented_101_epochs/overall_jitter_time_mag/model.pt"
        support_set_path = "support/cjy_01"
        net = triplet_network.TAPID_CNNEmbedding()
        net.load_state_dict(torch.load(model_path))
        net.eval()
        self.model = net
        support_size = 5
        support_include_all_conditions = True
        surface = "base"

        support_data = supportDataset.load_support_dataset(support_set_path, surface,
                                                           support_size, support_include_all_conditions)

        test_loader = DataLoader(support_data, batch_size=1, shuffle=False)
        with torch.no_grad():
            self.support = pd.DataFrame(
                [(net(i[0].unsqueeze(0))[0].numpy(), i[1].item()) for i in test_loader.dataset],
                columns=["embedding", "label"])
        self.knn_n = 7
        self.labels_name = support_data.labels
    def predict(self,input):
        # prepocess
        mean = [0.88370824, -1.0719419, 9.571041, -0.0018323545, -0.0061315685, -0.0150832655]
        std = [0.32794556, 0.38917893, 0.35336846, 0.099675156, 0.117989756, 0.06230596]
        input = ((input.squeeze().T - torch.tensor(mean)) / torch.tensor(std)).T.unsqueeze(0)

        with torch.no_grad():
            embedding =self.model(input)[0]
        scores = [ calc_distance(embedding,torch.from_numpy(i)) for i in self.support["embedding"]]

        scores = torch.stack(scores).squeeze()

        label,weight = knn(scores,self.support["label"].values,self.knn_n)
        pred_val = torch.tensor(label).unsqueeze(0)
        pred_label = self.labels_name[pred_val.item()]
        # if weight < 2:
        #     pred_label = "unknown"
        pred_label += f",{weight}"
        return pred_label


def calc_distance(x1,x2):
    # return (x1-x2).pow(2).sum(1)
    dist = nn.PairwiseDistance(p=2)
    return dist(x1,x2)

def knn(distance,labels,k):
    sorted_idx = torch.argsort(distance)[:k]
    df = pd.DataFrame({
        "distance":distance[sorted_idx],
        "labels":labels[sorted_idx],
    })
    df["weight"] = 1
    df = df.groupby("labels").sum().sort_values("weight",ascending=False)
    return df.index[0] , df.iloc[0]['weight']
if __name__ == '__main__':
    model_path = "../assets/res/final_result_margin_0.01/overall/model.pt"
    support_set_path = "../assets/input/cjy"
    classifier = TripletClassifier(model_path, support_set_path)
    classifier.predict(torch.randn(1,6,128))
    print(classifier.support)