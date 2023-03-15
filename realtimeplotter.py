from math import sin, cos

import matplotlib.pyplot
import matplotlib.pyplot as plt
import time


class RealTimePlotter:
    def __init__(self ,axis_num , x_length = 50):
        self.axis_num = axis_num
        self.length = x_length
        self.data = [ [] for i in range(axis_num)]
        plt.ion()
        self.figure,(self.ax1,self.ax2) = plt.subplots(2,1)
        # self.line, = self.ax.plot([],[])
        self.lines = []
        for i in range(3):
            self.lines.append(self.ax1.plot([], [])[0])
        for i in range(3):
            self.lines.append(self.ax2.plot([], [])[0])
        # self.ax1.set_autoscaley_on(True)
        self.ax1.set_xlim(0, x_length)
        self.ax1.set_ylim(-6,15)
        # self.ax1.set_autoscaley_on(True)
        self.ax2.set_xlim(0, x_length)
        self.ax2.set_ylim(-1, 1)
        self.count =0
        self.ax1.set_title("Acc")
        self.ax2.set_title("Gyro")
        self.ax1.legend(self.lines[:3], ["accX", "accY","accZ"], loc ='upper left')
        self.ax2.legend(self.lines[-3:], ["gyroX", "gyroY","gyroZ"], loc ='upper left')


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

        if self.count >=10:
            # self.ax1.relim()
            # self.ax1.autoscale_view()
            # self.ax2.relim()
            # self.ax2.autoscale_view()
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            self.count=0
            plt.pause(0.000001)


if __name__ == '__main__':
    plotter =  RealTimePlotter(6,100)
    for i in range(1000):
        plotter.add_data([2*j + (sin(i) if j <3 else cos(i)) for j in range(6)])
        print(plotter.data)

    pass