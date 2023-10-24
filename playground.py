import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import  random
import Utils


# def draw_signal(surface,gesture):

    # Utils.plot_data(,"",gesture,"",False)

def draw_resutl():
    x=["baseline","jitter","time","mag","jitter\n+time","jitter\n+mag","time\n+mag","jitter\n+time\n+mag"]
    y=[0.9355,0.9112,0.9383,0.9430,0.928,0.915,0.939,0.912]
    #y lim
    plt.figure(figsize=(10, 5))
    #plot baseline

    plt.hlines(0.9355,-1,len(y),colors="r",linestyles="dashed")
    plt.bar(x,y)
    plt.ylim(0.9,0.95)
    for i, v in enumerate(y):
        plt.text(i - 0.035 * len(x), v + 0.002, str(round(v, 3)))

    plt.show()

def show_different_sigma(tech,sigmas,results):
    plt.figure(figsize=(10, 5))
    plt.bar(sigmas,results)
    plt.ylim(0.85,0.95)

    plt.xlabel("Sigma")
    plt.title(tech)
    plt.hlines(0.9355,-0.5,len(results)-0.5,colors="r",linestyles="dashed")
    for i, v in enumerate(results):
        plt.text(i - 0.035 * len(sigmas), v + 0.002, str(round(v, 3)))
    plt.show()
# def detect_signal():

def draw_signals_on_different_surface(gesture,surfaces):
    data = []
    for surface in surfaces:
        data += [get_data(surface,gesture)]
    data = pd.concat(data)

    Utils.plot_data(data,"",gesture,"",False)

def draw_signal(gesture,surface,title):
    data = get_data(surface,gesture)[:150]
    Utils.plot_data(data,"mid_term",title,surface,True)
def get_data(surface,gesture):
    root = "assets/input/cjy"
    dir = os.path.join(root, surface, gesture)
    filename = random.choice(os.listdir(dir))
    data = pd.read_csv(os.path.join(dir,filename))
    return data

if __name__ == '__main__':
    # draw_signal("swipe_right","base","Swipe Right on Table")
    # draw_signal("swipe_right","wall","Swipe Right on Wall")
    draw_resutl()
    jitter={
        "baseline":0.9355,
        "0.01":0.936,
        "0.05":0.911,
        "0.1":0.925,
        "0.5":0.895
    }
    time={
        "baseline": 0.9355,
        "0.1": 0.913,
        "0.2": 0.938,
        "0.5": 0.926,
        "1": 0.918
    }
    mag={
        "baseline": 0.9355,
        "0.05": 0.932,
        "0.1": 0.943,
        "0.5": 0.909,
        "1": 0.876
    }

    show_different_sigma("jitter",jitter.keys(),jitter.values())
    show_different_sigma("time",time.keys(),time.values())
    show_different_sigma("mag",mag.keys(),mag.values())
