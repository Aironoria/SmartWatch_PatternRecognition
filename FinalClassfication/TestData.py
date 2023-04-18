
import  data
import os
import pandas as pd


def load_support_and_query_dataset(root, surface, length,include_all_conditions=False):
    include_new = False
    support_surface= []
    support_list = []
    if include_all_conditions:
        for dir in os.listdir(root):
            if dir != "new":
                support_surface.append(dir)
        if surface == "new" or include_new:
            support_surface.append("new")
    else:
        support_surface = [surface]
        if surface == "new":
            support_surface.append("base")
    for i, the_surface in enumerate(support_surface):
        df = []
        with open(os.path.join(root + "_train_test", the_surface, "train.txt"), 'r') as f:
            for line in f.readlines():
                gesture = line.split(os.sep)[0]
                df.append({
                    "path": os.path.join(root, the_surface, line),
                    "gesture": gesture
                })
        df = pd.DataFrame(df)
        support_list.extend(df.groupby("gesture").head(length)['path'])

    query_list = []
    with open(os.path.join(root + "_train_test", surface, "test.txt"), 'r') as f:
        for line in f.readlines():
            query_list.append(os.path.join(root, surface, line))
    if surface == "new":
        with open(os.path.join(root + "_train_test", "base", "test.txt"), 'r') as f:
            for line in f.readlines():
                query_list.append(os.path.join(root, "base", line))

    labels = list(set([i.split(os.sep)[-2] for i in query_list]))
    return data.FoodDataset(root, support_list, labels= labels), data.FoodDataset(root, query_list, labels= labels)