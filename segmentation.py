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



#find the segmentation
#1. find the peak
#2. validate if the peak is in a certain range


def calculate_energy(x,y,z,window_size=20,method=None):
    x = x[-window_size:]
    y = y[-window_size:]
    z = z[-window_size:]

    method = "diff"
    if method == "square_sum":
        return np.sum((x-x.mean())**2 +  (y - y.mean())**2 + (z - z.mean())**2)
    elif method == "diff":
        return np.abs(np.diff(x,n=2)).sum()+np.abs(np.diff(y,n=2)).sum()+np.abs(np.diff(z,n=2)).sum()
    elif method == "diff_std":
        return np.diff(np.sqrt(x**2+y**2+z**2)).std()
    else:
        return x.std() + y.std() + z.std()




def random_pick_signal(dataset="cjy",surface = None,gesture= None):
    root= os.path.join("assets","input",dataset)
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
    #padding df 128 lines at head with its first row and 128 lines at end with its final row
    # Concatenate the padding and original dataframes

    return df, os.path.join(surface,gesture,file)



def get_energies(x,y,z):
    energies = []
    for i in range(len(x)):
        if i == 5:
            a = calculate_energy(x[:i], y[:i], z[:i])
        energies.append( calculate_energy(x[:i], y[:i], z[:i]))

    return np.array(energies)

def find_peak(energies):
    peak_detection_window=50
    middle_index = (int)(peak_detection_window / 2)
    for threshold in [8,7,6]:
        for i in range(1, len(energies)):
            value = np.max(energies[max(i - peak_detection_window, 0):i])
            if energies[max(0, i - middle_index)] == value and value > threshold:
                return i - middle_index

    return energies.argmax()


def find_segment(df):
    length =128
    energies = get_energies(df['ax'], df['ay'], df['az'])
    peak = find_peak(energies)
    start_index = peak - 64
    if start_index < 0:
        start_index = 5
    elif start_index + length > len(df):
        start_index = len(energies) - length - 5
    return start_index
def scan_whole_dataset():
    root= os.path.join("assets","input","cjy_01")
    failed_list = []
    # create a cvs, column is filename and start point
    df = pd.DataFrame(columns=["path","start_point"])
    for person in os.listdir(root):
        if person == ".DS_Store":
            continue
        for gesture in os.listdir(os.path.join(root,person)):

            if gesture == "touchup" or gesture == "touchdown" or gesture == "nothing" or gesture == ".DS_Store":
                continue

            for filename in os.listdir(os.path.join(root,person,gesture)):
                if filename == ".DS_Store":
                    continue
                signal = pd.read_csv(os.path.join(root,person,gesture,filename))
                df.loc[len(df)] = [os.path.join(person,gesture,filename),find_segment(signal)]
    #save the failed list

    df.to_csv("segmentation_result_cjy_01.csv",index=False)
    with open("failed_list_cjy_new.txt","w") as f:
        for line in failed_list:
            f.write(line+"\n")

window_size = 20


def segment_one_and_draw(signal,filename):
    ax, ay, az = signal["ax"], signal["ay"], signal["az"]
    gx, gy, gz = signal["gx"], signal["gy"], signal["gz"]
    n_rows = 2
    n_cols = 2
    step_size = 1

    acc_energies = get_energies(ax, ay, az)
    peak = find_peak(acc_energies)
    start = find_segment(signal)
    end = start + 128
    plt.figure(figsize=(n_cols * 5, n_rows * 5))
    plt.suptitle(filename)
    plt.subplot(n_rows, n_cols, 1)
    plt.plot(ax)
    plt.plot(ay)
    plt.plot(az)
    plt.title("acc")
    plt.subplot(n_rows, n_cols, 1 + n_cols)
    plt.plot(gx)
    plt.plot(gy)
    plt.plot(gz)
    plt.title("gyro")

    plt.subplot(n_rows, n_cols, 2)
    plt.plot(acc_energies)
    plt.title("acc energy")

    print("peak", peak)
    print("start", start)
    print("end", end)
    print("shifted_peak",start+64)
    plt.scatter(peak, acc_energies[peak], c="r", s=100)
    plt.scatter(start, acc_energies[start], c="g", s=100)
    plt.scatter(end, acc_energies[end], c="g", s=100)
    plt.subplot(n_rows, n_cols, 2 + n_cols)
    plt.plot(get_energies(gx, gy, gz))
    plt.title("gyro energy")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
  # scan_whole_dataset()
  # signal, filename = random_pick_signal(dataset="cjy_01",gesture="pinch")
  # segment_one_and_draw(signal,filename)
  signal, filename = random_pick_signal(dataset="cjy_03", gesture="click")
  segment_one_and_draw(signal, filename)

  #random pick line from the failed list
  # with open("failed_list.txt","r") as f:
  #       lines = f.readlines()
  #       filename = random.choice(lines)
  #       signal = pd.read_csv(filename.strip())
  #       segment_one_and_draw(signal,filename.strip())