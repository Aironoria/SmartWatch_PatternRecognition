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

        color = [(255,0,0),(0,255,0),(0,0,255)]
        # color = ["r","g","b"]
        label = ["x","y","z"]
        acc_sub = self.plot_layout.addPlot(0,0)
        # negative: left,top ; positive: right, bottom
        acc_sub.addLegend(offset=(180, 1))
        self.acc_data = [np.zeros(600) for i in range(3)]
        self.acc_curve = [acc_sub.plot(self.acc_data[i], name="a"+label[i], pen = color[i]) for i in range(3)]
        # acc_sub.showLabel("left",show=True)
        self.acc_ptr = 0


        gyro_sub = self.plot_layout.addPlot(0,1)
        self.gyro_data = [np.zeros(600) for i in range(3)]
        self.gyro_curve = [gyro_sub.plot(self.gyro_data[i], name="mode" + str(i)) for i in range(3)]

        acc_energy_sub = self.plot_layout.addPlot(1, 0)
        self.acc_energy_data = np.zeros(600)
        self.acc_energy_curve = acc_energy_sub.plot(self.acc_energy_data)

        gyro_energy_sub = self.plot_layout.addPlot(1, 1)
        self.gyro_energy_data = np.zeros(600)
        self.gyro_energy_curve = gyro_energy_sub.plot(self.acc_energy_data)


    def connect(self,data_collector):
        data_collector.signal.connect(self.update_data)


    @pyqtSlot(np.ndarray)
    def update_data(self,data):
        self.acc_ptr += 1
        for i in range(3):
            self.acc_data[i][:-1] = self.acc_data[i][1:]
            self.acc_data[i][-1] = data[0][i]
            self.acc_curve[i].setData(self.acc_data[i])
            self.acc_curve[i].setPos(self.acc_ptr,0)

        for i in range(3):
            self.gyro_data[i][:-1] = self.gyro_data[i][1:]
            self.gyro_data[i][-1] = data[0][i]
            self.gyro_curve[i].setData(self.gyro_data[i])
            self.gyro_curve[i].setPos(self.acc_ptr,0)



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
        acc = self.encode_free_acceleration(data)
        self.signal.emit(acc)
        acc_characteristic_uuid = "0000fff1-0000-1000-8000-00805f9b34fb"
        if(time.time()-self.start_time>10):
            self.start_time = time.time()
            async with BleakClient(self.MODEL_NBR_UUID) as client:
                client.read_gatt_char(acc_characteristic_uuid)



    def encode_free_acceleration(self,bytes_):
        # These bytes are grouped according to Movella's BLE specification doc
        data_segments = np.dtype([
            ('x', np.double),
            ('y', np.double),
            ('z', np.double),
        ])
        formatted_data = np.frombuffer(bytes_, dtype=data_segments)
        return formatted_data


if __name__ == '__main__':
    app = QApplication(sys.argv)
    plotter = Plotter()
    data_collector = DataCollector()
    plotter.connect(data_collector)
    data_collector.start()
    print("B")
    sys.exit(app.exec_())

