import random

import numpy as np
from scipy.interpolate import CubicSpline      # for warping
import TripletLoss.triplet_data as triplet_data
import Utils
import matplotlib.pyplot as plt

import config

import os
import pandas as pd
def GenerateRandomCurves(X, sigma=0.2, knot=4,same_for_axis=True):
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cubic_splines = [ CubicSpline(xx[:,i],yy[:,i]) for i in range(X.shape[1])]
    return np.array([cubic_splines[0 if same_for_axis else  i](x_range) for i in range(X.shape[1])]).transpose()

def DistortTimesteps(X, sigma=0.2,same_for_axis = True):
    tt = GenerateRandomCurves(X, sigma,same_for_axis=same_for_axis) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale= [ (X.shape[0]-1)/tt_cum[-1,i] for i in range(X.shape[1])]
    for i in range(X.shape[1]):
        tt_cum[:,i] *= t_scale[i]

    return tt_cum





def TimeWarping(X,sigma=0.2,same_for_axis = True): #[N,channels]
    if not config.printed:
        config.printed = True
        print(f"Time: {sigma}")

    tt = DistortTimesteps(X, sigma,same_for_axis)
    X_new = np.zeros(X.shape)
    for i in range(X.shape[1]):
        X_new[:,i] = np.interp(np.arange(X.shape[0]), tt[:,i], X[:,i])
    return X_new

def MagnitudeWarping(X,sigma=0.1):
    if not config.printed:
        config.printed = True
        print(f"mag: {sigma}")

    return X * GenerateRandomCurves(X, sigma)
def Jitter(X, sigma=0.05):
    if not config.printed:
        config.printed = True
        print(f"Jitter: {sigma}")
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + myNoise


def draw_mag_warp(sample):
    curve = GenerateRandomCurves(sample, sigma= 0.2,knot=6)
    res = sample * curve

    plt.plot(np.arange(0, 100), np.ones(100), color="red", linestyle="--",label="original")
    plt.plot(np.arange(0, 100), curve, label="Warp Curve")
    plt.xlabel("Timestamp")
    plt.ylabel("Magnitude Factor")
    plt.legend()
    #lim
    plt.ylim([0, 1.5])
    plt.show()

    plt.plot(sample, label="original" , linestyle="--", color = "red")
    plt.plot(res, label="result")
    plt.xlabel("Timestamp")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.show()
def draw_time_warp(sample):
    tt = DistortTimesteps(sample, sigma=0.3)
    res = np.zeros(sample.shape)
    for i in range(sample.shape[1]):
        res[:, i] = np.interp(np.arange(sample.shape[0]), tt[:, i], sample[:, i])
    plt.plot(np.arange(0, 100), np.arange(0, 100),color="red",linestyle="--",label="original")
    plt.plot(np.arange(0, 100), tt,label="Warp Curve")
    plt.xlabel("Timestamp before warping")
    plt.ylabel("Timestamp after warping")
    plt.legend()
    plt.show()

    plt.plot(sample, label="original", linestyle="--", color="red",dashes=(5, 5))
    plt.plot(res, label="result")
    plt.xlabel("Timestamp")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.show()

def draw_original(sample):
     for i in range(sample.shape[1]):
         plt.plot(sample[:,i],label=i)
     plt.xlabel("Timestamp")
     plt.ylabel("Signal Value")
     plt.show()


def draw_jitter(sample):
    noise = np.random.normal(loc=0, scale=0.2, size=sample.shape)
    plt.plot(np.arange(0, 100),  noise, label="Noise")
    plt.ylim([-1, 1])
    plt.xlabel("Timestamp")
    plt.ylabel("Noise Value")
    plt.show()

    res = sample + noise
    plt.plot(sample, label="original", linestyle="--", color="red",dashes=(5, 5))
    plt.plot(res, label="result")
    plt.xlabel("Timestamp")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.show()

import matplotlib.patches as patches
def draw_slice(sample):
    #draw three rect showing different slice, the width of rect is 64, the height is 5
    #the distance between two rect is 5
    #the first rect is at 10
    #the color of rect is using [red,green,blue]

    plt.plot(sample)
    plt.xlabel("Timestamp")
    plt.ylabel("Signal Value")
    base = -3
    up = 0.1
    #draw rect
    rect = patches.Rectangle((10, base), 64, 5, linewidth=2, edgecolor='r', facecolor='red',label="slice 1",alpha=0.1)
    plt.gca().add_patch(rect)
    rect = patches.Rectangle((13, base+up ), 64, 5, linewidth=2, edgecolor='g', facecolor='green',label="slice 2",alpha=0.1)
    plt.gca().add_patch(rect)
    rect = patches.Rectangle((16, base+up*2), 64, 5, linewidth=2, edgecolor='b', facecolor='blue',label="slice 3",alpha=0.1)
    plt.gca().add_patch(rect)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    dir = os.path.join("assets","input","cjy","base","swipe_right")
    filename = random.choice(os.listdir(dir))
    # filename = "11_06_18_07_39.csv"
    filename ="11_06_18_05_06.csv"
    print(filename)
    df = pd.read_csv(os.path.join(dir,filename))
    sample = df.values[25:25+100,1:2] +2

    print(sample.shape)


    draw_slice(sample)
    # draw_time_warp(sample)
    # draw_mag_warp(sample)
    # ax = fig.add_subplot(211)
    # ax.set_ylim([-10, 10])

    # for i in range(sample.shape[1]):
    #     plt.plot(sample[:,i],label=i)

    # ax = fig.add_subplot(212)
    # # ax.set_ylim([-10, 10])
    #
    # for i in range(after_sample.shape[1]):
    #     plt.plot(after_sample[:, i], label=i)
