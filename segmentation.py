#find the beginning and ending timestamp of the signal
import os
import random
import pandas as pd
import numpy as np
import Utils
import matplotlib.pyplot as plt

#1. random choice a signal
#2. find the beginning and ending timestamp of the signal
#3. plot the engergy result
#4. plot the timestamp result


def rcs(data):
    pass
def calculate_energy(data):
    pass
def calculate_standard_deviation(acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z):
    return [acc_x.std()+acc_y.std()+acc_z.std(), gyro_x.std()+gyro_y.std()+gyro_z.std()]

def calculate_square_sum(acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z):
    return [np.sum((acc_x-acc_x.mean())**2 +  (acc_y - acc_y.mean())**2 + (acc_z - acc_z.mean())**2),
            np.sum((gyro_x-gyro_x.mean())**2 +  (gyro_y - gyro_y.mean())**2 + (gyro_z - gyro_z.mean())**2)]
def calculate_differential(acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z):
    return [acc_x.diff().abs().sum()+acc_y.diff().abs().sum()+acc_z.diff().abs().sum(),
            gyro_x.diff().abs().sum()+gyro_y.diff().abs().sum()+gyro_z.diff().abs().sum()]

def calculate_differential_and_std(acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z):
    return [np.sqrt(acc_x**2+acc_y**2+acc_z**2).diff().std(),
            np.sqrt(gyro_x**2+gyro_y**2+gyro_z**2).diff().std()]
def segment_signal(signal,window_size=20,step_size=10,method=None):
    res = []
    for i in range(window_size):
        res.append([0,0])
    for i in range(0,len(signal)-window_size,step_size):
        signal_window = signal[i:i+window_size]
        acc_x = signal_window["ax"]
        acc_y = signal_window["ay"]
        acc_z = signal_window["az"]
        gyro_x = signal_window["gx"]
        gyro_y = signal_window["gy"]
        gyro_z = signal_window["gz"]
        if method =="std":
            processed_window = calculate_standard_deviation(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
        elif method =="square_sum":
            processed_window = calculate_square_sum(acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z)
        elif method =="diff":
            processed_window = calculate_differential(acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z)
        elif method =="diff_std":
            processed_window = calculate_differential_and_std(acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z)
        else:
            processed_window = calculate_standard_deviation(acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z)
        for _ in range(step_size):
            res.append(processed_window)
    res = np.array(res)

    return res





def random_pick_signal(surface = None,gesture= None):
    root= os.path.join("assets","input","ten_data")
    if surface is None:
        surface = random.choice(os.listdir(root))
    root = os.path.join(root,surface)
    if gesture is None:
        while True:
            gesture = random.choice(os.listdir(os.path.join(root)))
            if gesture != "touchup" and gesture != "touchdown":
                break
    file = random.choice(os.listdir(os.path.join(root,gesture)))
    df = pd.read_csv(os.path.join(root,gesture,file))
    return df, os.path.join(surface,gesture,file)

if __name__ == '__main__':
    # signal, filename = random_pick_signal("quyuqi","nothing")
    signal, filename = random_pick_signal()
    # Utils.plot_data(signal,"",filename,"",False)


    n_rows= 2
    n_cols = 3
    window_size = 50
    step_size = 10
    res =segment_signal(signal,window_size,step_size,"diff")

    plt.figure(figsize=(n_cols*5,n_rows*5))
    plt.suptitle(filename)
    plt.subplot(n_rows,n_cols,1)
    plt.plot(signal["ax"])
    plt.plot(signal["ay"])
    plt.plot(signal["az"])
    plt.title("acc")
    plt.subplot(n_rows,n_cols,1+n_cols)
    plt.plot(signal["gx"])
    plt.plot(signal["gy"])
    plt.plot(signal["gz"])
    plt.title("gyro")

    res =segment_signal(signal,window_size,step_size,"std")
    plt.subplot(n_rows,n_cols,2)
    plt.plot(res[:,0])
    plt.title("acc std")
    plt.subplot(n_rows,n_cols,2+n_cols)
    plt.plot(res[:,1])
    plt.title("gyro std")

    res =segment_signal(signal,window_size,step_size,"diff_std")
    plt.subplot(n_rows,n_cols,3)
    plt.plot(res[:,0])
    plt.title("acc diff")
    plt.subplot(n_rows,n_cols,3+n_cols)
    plt.plot(res[:,1])
    plt.title("gyro diff")

    plt.tight_layout()
    plt.show()
