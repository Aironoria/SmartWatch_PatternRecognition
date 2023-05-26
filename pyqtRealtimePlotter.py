
import sys
from PyQt5.Qt import *
from pyqtgraph import PlotWidget
from PyQt5 import QtCore
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
import pyqtgraph.examples

class Plotter(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(600, 600)
        # 添加 PlotWidget 控件
        self.plotWidget_ted = PlotWidget(self)
        # 设置该控件尺寸和相对位置
        self.plotWidget_ted.setGeometry(QtCore.QRect(25, 25, 550, 550))

        # 仿写 mode1 代码中的数据
        # 生成 300 个正态分布的随机数
        self.data1 = np.zeros(300)

        self.curve2 = self.plotWidget_ted.plot(self.data1, name="mode2")
        self.ptr1 = 0


    # 数据左移
    def update_data(self,data):
        self.data1[:-1] = self.data1[1:]
        self.data1[-1] = data
        # 数据填充到绘制曲线中
        self.curve2.setData(self.data1)
        # x 轴记录点
        self.ptr1 += 1
        # 重新设定 x 相关的坐标原点
        # self.curve2.setPos(self.ptr1,0)





if __name__ == '__main__':
    example = pg.examples.run()