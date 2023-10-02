import pandas as pd
from classifier import TripletClassifier
import torch
import time

if __name__ == '__main__':
    df = pd.read_csv("record.csv")
    window_size = 128
    half_window=64
    classifier = TripletClassifier()
    start = time.time()
    for i in range(0,len(df)-window_size):
        window = df[i:i+window_size]
        data = torch.tensor(window[["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]].values, dtype=torch.float32).T
        label = classifier.predict(data)
        peak = i +half_window
        df.loc[peak-1,"off_line_label"] = label + f"[{peak+300 -half_window}:{peak+300+half_window}]"
    df.to_csv("record_with_offline.csv",index=False)
    print(time.time()-start)
