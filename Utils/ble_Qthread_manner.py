import asyncio
import sys
import threading
import time
import pyqtgraph as pg

from PyQt5.Qt import *
import numpy as np
from bleak import BleakClient, BleakScanner

from RealTime.Filter import ButterWorthBandpassFilter
from classifier import TripletClassifier
import torch
import pandas as pd
#should implement new function: add samples

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
        self.acc_peaks = acc_energy_sub.plot([],[],pen=None,symbol='o',symbolPen='r',symbolSize=5,symbolBrush=0.2)
        self.acc_energy_sub = acc_energy_sub


        gyro_energy_sub = self.plot_layout.addPlot(1, 1)
        self.gyro_energy_data = np.zeros(x_lim)
        self.gyro_energy_curve = gyro_energy_sub.plot(self.acc_energy_data)

        self.predicted_results = []


        self.classifier = TripletClassifier()
        self.record = pd.DataFrame(columns=["timestamp","acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z","label"])
        self.filter =  ButterWorthBandpassFilter(5, 32, 100, order=5)


    def connect(self,data_collector):
        data_collector.signal.connect(self.update_data)

    def classify(self,peak):
        peak = peak - self.acc_ptr -25
        half_window=64
        data = np.array([self.acc_data[0][peak-half_window:peak+half_window],
                self.acc_data[1][peak-half_window:peak+half_window],
                self.acc_data[2][peak-half_window:peak+half_window],
                self.gyro_data[0][peak-half_window:peak+half_window],
                self.gyro_data[1][peak-half_window:peak+half_window]-1,
                self.gyro_data[2][peak-half_window:peak+half_window]-2])
        data =torch.tensor(data,dtype=torch.float32)
        data = data.unsqueeze(0)
        res = self.classifier.predict(data)
        peak = peak + self.acc_ptr
        self.record.loc[self.record["timestamp"] == peak,"label"] = res+ f"[{peak-half_window}:{peak+half_window}]"
        return res
    def plot_acc_peaks(self,x,y): #peak detected
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

        label = self.classify(x)
        predicted_result = pg.TextItem(text=label)
        predicted_result.setPos(x,y)
        predicted_result.setColor('r')
        self.acc_energy_sub.addItem(predicted_result)
        self.predicted_results.append(predicted_result)






    def detect_acc_peaks(self):
        x_array, y_array = self.acc_peaks.getData()
        if x_array is not None and x_array[0]< self.acc_ptr:  # remove the peaks that are out of the window
            index = np.where(x_array > self.acc_ptr)
            x_array = x_array[index]
            y_array = y_array[index]
            self.acc_peaks.setData(x_array, y_array)
            label_need_to_remove = self.predicted_results[0]
            self.acc_energy_sub.removeItem(label_need_to_remove)
            self.predicted_results.remove(label_need_to_remove)

        current = self.acc_ptr + self.x_lim

        if current < 400:
            return
        detecting_window = 128

        # find the maxmium in the last 50 points in the acc_energy_data

        value = np.max(self.acc_energy_data[-detecting_window:])
        middle_index = (int)(detecting_window / 2)
        if self.acc_energy_data[-middle_index] == value: #find a peak
            # if self.acc_energy_data[-(detecting_window)]<valu*3/4 and self.acc_energy_data[-1]<value*3/4: #valid peak
                if  1.2 <value < 8: #valid peak
                    if self.acc_peaks.getData()[0] is None:
                        last_peak = 0
                    else:
                        last_peak = self.acc_peaks.getData()[0][-1]
                    if current - last_peak > 130:
                        # print(f"first {self.acc_energy_data[-detecting_window] }, middle {value}, end {self.acc_energy_data[-1]}")
                        self.plot_acc_peaks( current - middle_index, value)
    @pyqtSlot(np.ndarray)
    def update_data(self,data):
        data = np.array([i for i in data[0]])
        data = self.filter.filter(data)
        self.acc_ptr += 1
        for i in range(3):
            self.acc_data[i][:-1] = self.acc_data[i][1:]
            self.acc_data[i][-1] = data[i]
            self.acc_curve[i].setData(self.acc_data[i])
            self.acc_curve[i].setPos(self.acc_ptr,0)
        self.acc_energy_data[:-1] = self.acc_energy_data[1:]
        self.acc_energy_data[-1] = segment_signal(self.acc_data[0],self.acc_data[1],self.acc_data[2],50)
        self.acc_energy_curve.setData(self.acc_energy_data)
        self.acc_energy_curve.setPos(self.acc_ptr,0)



        for i in range(3):
            self.gyro_data[i][:-1] = self.gyro_data[i][1:]
            self.gyro_data[i][-1] = data[i+3]+i
            self.gyro_curve[i].setData(self.gyro_data[i])
            self.gyro_curve[i].setPos(self.acc_ptr,0)

        self.gyro_energy_data[:-1] = self.gyro_energy_data[1:]
        self.gyro_energy_data[-1] = segment_signal(self.gyro_data[0],self.gyro_data[1],self.gyro_data[2],window_size=20)
        self.gyro_energy_curve.setData(self.gyro_energy_data)
        self.gyro_energy_curve.setPos(self.acc_ptr,0)

        self.record = self.record.append({
            "timestamp": self.acc_ptr+self.x_lim,
            "acc_x": data[0],
            "acc_y": data[1],
            "acc_z": data[2],
            "gyro_x": data[3],
            "gyro_y": data[4],
            "gyro_z": data[5],
        },ignore_index=True)
        self.detect_acc_peaks()

        if self.acc_ptr == 2900:
            print("saving")
            self.record.to_csv("record.csv",index=False)
            print("saved")



class DataCollector(QThread):
    signal = pyqtSignal(np.ndarray)
    def __init__(self):
        super().__init__()



    def run(self):
       asyncio.run(self.task())
    async def task(self):
        acc_characteristic_uuid = "0000fff1-0000-1000-8000-00805f9b34fb"
        firmware_characteristic_uuid = "00002a26-0000-1000-8000-00805f9b34fb"
        self.MODEL_NBR_UUID = await self.scan()

        async with BleakClient(self.MODEL_NBR_UUID) as client:
            # asyncio.create_task(self.notification_task(client, acc_characteristic_uuid))
            await client.start_notify(acc_characteristic_uuid, self.notification_callback)
            # update_interval = 35  # Set your desired read interval (in seconds)
            while True:
                await asyncio.sleep(1)
            #     await client.write_gatt_char(firmware_characteristic_uuid, bytearray([]))

    async def notification_task(self, client, acc_characteristic_uuid):
        await client.start_notify(acc_characteristic_uuid, self.notification_callback)


    async def read(self, client):
        acc_characteristic_uuid = "0000fff1-0000-1000-8000-00805f9b34fb"
        print("read")
        res = await client.write_gatt_char(acc_characteristic_uuid, bytearray([]))

        print(res)
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

    def notification_callback(self,sender, data):
        global  count, plotter
        acc_gyro_data = self.encode_free_acceleration(data)
        self.signal.emit(acc_gyro_data)



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

