import asyncio
import sys
import time
import pyqtgraph as pg
from PyQt5.Qt import *
from pyqtgraph import PlotWidget
from PyQt5 import QtCore
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
from bleak import BleakClient, BleakScanner

class Plotter(QWidget):
    def __init__(self):
        super().__init__()
        # 添加 PlotWidget 控件
        self.plot_layout = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples",size=(600 ,600))
        x_lim = 300
        self.x_lim = x_lim
        color = [(255,0,0),(0,255,0),(0,0,255)]
        # color = ["r","g","b"]
        label = ["x","y","z"]
        acc_sub = self.plot_layout.addPlot(0,0)
        # negative: left,top ; positive: right, bottom
        acc_sub.addLegend(offset=(180, 1))
        self.acc_data = [np.zeros(x_lim) for i in range(3)]
        self.acc_curve = [acc_sub.plot(self.acc_data[i], name="a"+label[i], pen = color[i]) for i in range(3)]
        # acc_sub.showLabel("left",show=True)
        self.acc_ptr = 0



        gyro_sub = self.plot_layout.addPlot(0,1)
        gyro_sub.addLegend(offset=(180, 1))
        self.gyro_data = [np.zeros(x_lim) for i in range(3)]
        self.gyro_curve = [gyro_sub.plot(self.gyro_data[i], name="g" +label[i], pen =color[i]) for i in range(3)]

        acc_energy_sub = self.plot_layout.addPlot(1, 0)
        self.acc_energy_data = np.zeros(x_lim)
        self.acc_energy_curve = acc_energy_sub.plot(self.acc_energy_data)
        self.acc_peaks = acc_energy_sub.plot([0],[0],pen=None,symbol='o',symbolPen='r',symbolSize=5,symbolBrush=0.2)


        gyro_energy_sub = self.plot_layout.addPlot(1, 1)
        self.gyro_energy_data = np.zeros(x_lim)
        self.gyro_energy_curve = gyro_energy_sub.plot(self.acc_energy_data)


    def connect(self,data_collector):
        data_collector.signal.connect(self.update_data)

    def plot_acc_peaks(self,x,y):
        x_array,y_array = self.acc_peaks.getData()
        if x_array is None:
            x_array = np.array([x])
            y_array = np.array([y])
        else:
            x_array = np.append(x_array,x)
            y_array = np.append(y_array,y)
            index  = np.where(x_array>self.acc_ptr)
            x_array = x_array[index]
            y_array = y_array[index]
        self.acc_peaks.setData(x_array,y_array)

    def detect_acc_peaks(self):

        current = self.acc_ptr + self.x_lim
        detecting_window = 50

        # find the maxmium in the last 50 points in the acc_energy_data

        value = np.max(self.acc_energy_data[-detecting_window:])
        middle_index = (int)(detecting_window / 2)
        if self.acc_energy_data[-middle_index] == value: #find a peak
            if  0.5 <value < 10: #valid peak
                last_peak = self.acc_peaks.getData()[0][-1]
                if current - last_peak > 128:
                    self.plot_acc_peaks( current - middle_index, value)
    @pyqtSlot(np.ndarray)
    def update_data(self,data):
        self.acc_ptr += 1
        for i in range(3):
            self.acc_data[i][:-1] = self.acc_data[i][1:]
            self.acc_data[i][-1] = data[0][i]
            self.acc_curve[i].setData(self.acc_data[i])
            self.acc_curve[i].setPos(self.acc_ptr,0)
        self.acc_energy_data[:-1] = self.acc_energy_data[1:]
        self.acc_energy_data[-1] = segment_signal(self.acc_data[0],self.acc_data[1],self.acc_data[2],50)
        self.acc_energy_curve.setData(self.acc_energy_data)
        self.acc_energy_curve.setPos(self.acc_ptr,0)



        for i in range(3):
            self.gyro_data[i][:-1] = self.gyro_data[i][1:]
            self.gyro_data[i][-1] = data[0][i+3]+i
            self.gyro_curve[i].setData(self.gyro_data[i])
            self.gyro_curve[i].setPos(self.acc_ptr,0)

        self.gyro_energy_data[:-1] = self.gyro_energy_data[1:]
        self.gyro_energy_data[-1] = segment_signal(self.gyro_data[0],self.gyro_data[1],self.gyro_data[2],window_size=20)
        self.gyro_energy_curve.setData(self.gyro_energy_data)
        self.gyro_energy_curve.setPos(self.acc_ptr,0)

        self.detect_acc_peaks()




class DataCollector(QThread):
    signal = pyqtSignal(np.ndarray)
    def __init__(self):
        super().__init__()



    def run(self):
        asyncio.run(self.task())

    async def task(self):
        acc_characteristic_uuid = "0000fff1-0000-1000-8000-00805f9b34fb"
        MODEL_NBR_UUID = "B929F36D-81A7-C6ED-76A0-BDFCE5B93E66"
        self.MODEL_NBR_UUID = await self.scan()

        async with BleakClient(self.MODEL_NBR_UUID) as client:
            print(client)  # prints True or False
            self.start_time = time.time()
            await client.start_notify(acc_characteristic_uuid, self.notification_callback)
    async def scan(self):
        print("scan")
        res = ""
        while True:
            devices = await BleakScanner.discover()
            for device in devices:
                # print(device)
                if device.name == "Galaxy Watch4 (SAJZ)":
                    res = device.address
            if res == "":
                print("No device found")
            else:
                print(res)
                return res

    async def notification_callback(self,sender, data):
        global start_time, count, plotter
        acc_gyro_data = self.encode_free_acceleration(data)
        self.signal.emit(acc_gyro_data)
        acc_characteristic_uuid = "0000fff1-0000-1000-8000-00805f9b34fb"
        if(time.time()-self.start_time>10):
            self.start_time = time.time()
            async with BleakClient(self.MODEL_NBR_UUID) as client:
                client.read_gatt_char(acc_characteristic_uuid)



    def encode_free_acceleration(self,bytes_):
        # These bytes are grouped according to Movella's BLE specification doc
        data_segments = np.dtype([
            ('ax', np.double),
            ('ay', np.double),
            ('az', np.double),
            ('gx', np.double),
            ('gy', np.double),
            ('gz', np.double)
        ])
        formatted_data = np.frombuffer(bytes_, dtype=data_segments)
        return formatted_data


def segment_signal(x,y,z,window_size=20,method=None):
    x = x[-window_size:]
    y = y[-window_size:]
    z = z[-window_size:]

    if method == "square_sum":
        return np.sum((x-x.mean())**2 +  (y - y.mean())**2 + (z - z.mean())**2)
    elif method == "diff":
        return np.abs(np.diff(x,n=2)).sum()+np.abs(np.diff(y,n=2)).sum()+np.abs(np.diff(z,n=2)).sum()
    elif method == "diff_std":
        return np.sqrt(x**2+y**2+z**2).diff().std()
    else:
        return x.std() + y.std() + z.std()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    plotter = Plotter()
    data_collector = DataCollector()
    plotter.connect(data_collector)
    data_collector.start()
    print("B")
    sys.exit(app.exec_())

