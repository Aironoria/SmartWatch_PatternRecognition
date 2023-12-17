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
    # plt.hlines(list(y)[0],-1,len(y),colors="r",linestyles="dashed")
    plt.bar(x,y,width=3)
    plt.ylim(0.5,1)
    #plt text on bar
    for i, v in enumerate(y):
        plt.text(5*i - 2.3 * len(x), v + 0.002, str(round(v, 3)))

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


def draw_three_bar(categories,values1,values2,baseline,tittle):


    # 柱状图的位置和宽度
    x = np.arange(len(categories))
    width = 0.25

    # 绘制柱状图
    plt.bar(x - width, values1, width, label='Conv_3')
    plt.bar(x, values2, width, label='Conv_4')
    # plt.bar(x + width, values3, width, label='VConv_5')
    plt.hlines(baseline, -1, len(values1), colors="r", linestyles="dashed")
    # 添加刻度标签和标题
    plt.xticks(x, categories)
    plt.ylabel('Values')
    plt.title(tittle)
    plt.ylim(0.7,0.95)
    # 添加图例
    plt.legend()

    # 显示图表
    plt.show()
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

    cjy_03 = {
        "-20":0.848,
        "-15":0.864,
        "-10":0.904,
        "-5":0.832,
        "0":0.848,
        "5":0.864,
        "10":0.824,
        "15":0.872,
        "20":0.776,
    }

    ljd = {
        "-20": 0.848,
        "-15": 0.864,
        "-10": 0.904,
        "-5": 0.832,
        "0": 0.848,
        "5": 0.864,
        "10": 0.824,
        "15": 0.872,
        "20": 0.776,
    }


    cjy_03 ={
        # -20: 0.8,
        -15: 0.824,
        -10: 0.824, -5: 0.848, 0: 0.848, 5: 0.88, 10: 0.904, 15: 0.92,
        # 20: 0.904
    }
    # draw_result(cjy_03.keys(),cjy_03.values())
    ljd = {-15: 0.816, -10: 0.8, -5: 0.856, 0: 0.84, 5: 0.848, 10: 0.8, 15: 0.792}
    draw_result(ljd.keys(),ljd.values())
    # draw_result(original.keys(),original.values())
    # draw_result(real_use_different_margin.keys(),real_use_different_margin.values())
    # draw_result(real_use_different_offset.keys(),real_use_different_offset.values())

    categories = ['64','80','100','128']
    conv3_cjy = [0.848, 0.864,0.904,0.832]
    conv_4_cjy =[0.864,0.864,0.824,0.872]
    conv_5_cjy =[0,0,0,0.776]


    conv3_ljd = [0.84, 0.72,0.752,0.768]
    conv_4_ljd =[0.784,0.728,0.752,0.792]
    conv_5_ljd =[0,0,0,0.76]
    # draw_three_bar(categories,conv3_cjy,conv_4_cjy,0.776,"P1")
    # draw_three_bar(categories,conv3_ljd,conv_4_ljd,0.76,"P2")
    # show_different_sigma("jitter",jitter.keys(),jitter.values())
    # show_different_sigma("time",time.keys(),time.values())
    # show_different_sigma("mag",mag.keys(),mag.values())
