import os.path

import numpy as np
import pandas as pd
import  random
import Utils


# def draw_signal(surface,gesture):

    # Utils.plot_data(,"",gesture,"",False)



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
    draw_signal("swipe_right","base","Swipe Right on Table")
    # draw_signal("swipe_right","wall","Swipe Right on Wall")