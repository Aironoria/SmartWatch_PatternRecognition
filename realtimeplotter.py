from math import sin

import matplotlib.pyplot
import matplotlib.pyplot as plt
import time


class RealTimePlotter:
    def __init__(self ,axis_num , x_length = 50):
        self.axis_num = axis_num
        self.length = x_length
        self.data = [ [] for i in range(axis_num)]
        plt.ion()
        self.figure,self.ax = plt.subplots()
        # self.line, = self.ax.plot([],[])
        self.lines = [ self.ax.plot([],[]) [0]for i in range(axis_num)]
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(0, x_length)
        self.count =0

    def show(self):
        pass

    def add_data(self,data):

        for i in range(self.axis_num):
            self.data[i].append(float(data[i]))
            self.data[i] = self.data[i][-self.length:]
            self.lines[i].set_xdata(range(len(self.data[i])))
            self.lines[i].set_ydata(self.data[i])
        self.count+=1
        # x = range(len(self.data[0]))
        # for i in range(self.axis_num):
        #     self.lines[i].set_xdata(range(len(self.data[0])))
        #     self.lines[i].set_ydata(self.data[i])

        if self.count >=5:
            self.ax.relim()
            self.ax.autoscale_view()
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            self.count=0


if __name__ == '__main__':
    plotter =  RealTimePlotter(6,100)
    for i in range(1000):
        plotter.add_data([2*j + sin(i) for j in range(6)])
        # print(plotter.data)

    pass