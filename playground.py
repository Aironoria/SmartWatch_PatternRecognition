import random

import numpy as np
import matplotlib.pyplot as plt

def plot(x,y):
    plt.figure()
    plt.plot(x, y)
    plt.show()

def sin_fft():
    sample_rate = 100
    t = np.arange(100) * 1 / sample_rate

    y = np.sin(30 * 2 * np.pi * t) + np.sin(15 * 2 * np.pi * t)

    for i in range(50, 100):
        y[i] = 1
    y_fft = np.fft.fft(y)
    y_fft = np.abs(y_fft)
    frequency = np.fft.fftfreq(len(y), 1 / sample_rate)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title("signal")
    plt.plot(t, y)
    plt.subplot(2, 1, 2)
    plt.title("FFT")
    plt.plot(frequency[frequency > 0], np.abs(y_fft)[frequency > 0])
    plt.show()

def fft(data,sample_rate):
    data_fft = np.fft.fft(data)
    frequency = np.fft.fftfreq(len(data), 1 / sample_rate)
    return frequency[frequency > 0], np.abs(data_fft)[frequency > 0]

def insert(dataframe, data, start_index , factor =1):
    res = dataframe.copy()
    for index in start_index:
        for i in range(len(data)):
            res[index+i] = data[i] * factor
    return res

def peak_index(data,distance =2 ,baseline =3):
    res =[]
    for i in range(distance, len(data)-distance):
        max = True
        if data[i] <baseline:
            max =False
            continue
        for j in range(1,distance+1):
            if data[i] < data[i+j] or data[i] < data[i-j]:
                max =False;break
        if max:
            res.append(i)
    return res


def plot_fft(data,sample_rate):

    plt.figure()
    plt.subplot(2,1,1)
    plt.title("signal")
    plt.plot(np.arange(len(data)),data)
    plt.subplot(2,1,2)
    plt.title("fft")
    fre, y_fft= fft(data,sample_rate)
    index = peak_index(y_fft)
    plt.plot(fre,y_fft)
    plt.plot(fre[index],y_fft[index],"r.")
    for i in index:
        plt.text(fre[-5], i*0.5 ,str(fre[i])+"hz" )
        plt.text(fre[i], y_fft[i], str(round(y_fft[i],1)))
    plt.subplots_adjust(hspace=0.5)
    plt.show()
def draw():
    sample_rate = 100
    N = 200
    t = np.arange(N) * 1 / sample_rate
    y = np.random.uniform(-0.1,0.1,N)
    d = [0.1,0.14,0.24,0.28,0.34, 0.4,0.41,0.37 ,0.35,0.2,0.15, 0.1,0.07,0,-0.1,-0.2,-0.22,-0.19,-0.18,-0.15,-0.10,-0.05,0]
    # for i in range(50,70):
    #     y[i] *= random.random() *30
    # for i in range(120,150):
    #     y[i] *= random.random() *30

    plot_fft(insert(y,d,[50,100]),sample_rate)

    # insert(y,d,140)
    # plot_fft(y,sample_rate)


    plot_fft(insert(
        insert(y,d,[50]) ,d, [100,150], 0.5
    ) , sample_rate)

    plot_fft(insert(y,d,[50,100,150]),sample_rate)

    plot_fft(insert(y,d,[30,150]),sample_rate)

    plot_fft(insert(y,d,[100,150]),sample_rate)

    plot_fft(insert(y,d,[70,150]),sample_rate)
#
# data = np.arange(49)
# res = np.fft.fft(data)
# res_abs = np.abs(res)
# breakpoint()


print(x_axis)
