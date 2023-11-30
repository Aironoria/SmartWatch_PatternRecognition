import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import  random
import Utils


# def draw_signal(surface,gesture):

    # Utils.plot_data(,"",gesture,"",False)

def draw_result(x,y):

    #y lim
    plt.figure(figsize=(10, 5))
    #plot baseline
    #the first value of y
    plt.hlines(list(y)[0],-1,len(y),colors="r",linestyles="dashed")
    plt.bar(x,y)
    plt.ylim(0.5,1)
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
    x = ["baseline", "jitter", "time", "mag", "jitter\n+time", "jitter\n+mag", "time\n+mag", "jitter\n+time\n+mag"]
    y = [0.9355, 0.9112, 0.9383, 0.9430, 0.928, 0.915, 0.939, 0.912]
    #using x and y to make a dict
    original = {
        "baseline":0.9355,
        "jitter":0.9112,
        "time":0.9383,
        "mag":0.9430,
        "jitter\n+time":0.928,
        "jitter\n+mag":0.915,
        "time\n+mag":0.939,
        "jitter\n+time\n+mag":0.912
    }
    original_cjy = {
        "baseline": 0.922,
        "jitter": 0.927,
        "time": 0.914,
        "mag": 0.916,
        "jitter\n+time": 0.930,
        "jitter\n+mag": 0.926,
        "time\n+mag": 0.939,
        "jitter\n+time\n+mag": 0.958
    }


    reail_time_segmentation = {
        "baseline": 0.964,
        "jitter": 0.930,
        "time": 0.945,
        "mag": 0.944,
        "jitter\n+time": 0.931,
        "jitter\n+mag": 0.945,
        "time\n+mag": 0.939,
        "jitter\n+time\n+mag": 0.933
    }

    reail_time_segmentation_cjy = {
        "baseline": 0.942,
        "jitter": 0.941,
        "time": 0.942,
        "mag": 0.934,
        "jitter\n+time": 0.937,
        "jitter\n+mag": 0.941,
        "time\n+mag": 0.922,
        "jitter\n+time\n+mag": 0.939
    }


    real_use_different_margin={
        "0.01\n baseline": 0.808,
        "0.1":0.776,
        "1": 0.832,
        "2":0.808,
        "5":0.752,
    }

    real_use_different_offset={
        "0\n baseline": 0.808,

        "5":0.848,
        "15": 0.864,
        "25":0.824,
        "30":0.816,
    }
    draw_result(real_use_different_margin.keys(),real_use_different_margin.values())
    draw_result(real_use_different_offset.keys(),real_use_different_offset.values())


    # show_different_sigma("jitter",jitter.keys(),jitter.values())
    # show_different_sigma("time",time.keys(),time.values())
    # show_different_sigma("mag",mag.keys(),mag.values())
