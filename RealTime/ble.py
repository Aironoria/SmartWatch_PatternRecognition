# https://medium.com/@protobioengineering/how-to-stream-data-from-a-movella-dot-wearable-sensor-with-a-mac-and-python-5822e76fb43e
import asyncio
import sys
from PyQt5.QtWidgets import QApplication
from bleak import BleakClient, BleakScanner
import numpy as np
import time
import pyqtRealtimePlotter
from threading import Thread
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph import QtWidgets


def notification_callback(sender, data):
    global start_time,count,plotter
    acc = encode_free_acceleration(data)
    # print(acc)
    if plotter is None:
        # print("plotter is none")
        return
    plotter.update_data(acc[0][0])
    # if start_time == 0:
    #     start_time = time.time()
    # elif time.time() - start_time < 1:
    #     count+=1
    # else:
    #     print(count)
    #     count = 0
    #     start_time = time.time()


def encode_free_acceleration(bytes_):
    # These bytes are grouped according to Movella's BLE specification doc
    data_segments = np.dtype([
        ('x', np.double),
        ('y', np.double),
        ('z', np.double),
        ])
    formatted_data = np.frombuffer(bytes_, dtype=data_segments)
    return formatted_data

async def scan():
    res = ""
    devices = await BleakScanner.discover()
    for device in devices:
        # print(device)
        if device.name =="Galaxy Watch4 (SAJZ)":
            res = device.address
    if res == "":
        print("No device found")
    else:
        print(res)
    return res
async def main():
    MODEL_NBR_UUID = await scan()

    async with BleakClient(MODEL_NBR_UUID) as client:
        print(client) # prints True or False
        a = await client.start_notify(acc_characteristic_uuid, notification_callback)
        print(a)
def thread_task():
    asyncio.run(main())

if __name__ == '__main__':
    acc_characteristic_uuid = "0000fff1-0000-1000-8000-00805f9b34fb"
    start_time = 0
    count = 0
    plotter =None
    thread = Thread(target=thread_task, args=())
    thread.start()
    print("A")
    app = QApplication(sys.argv)
    plotter = pyqtRealtimePlotter.Plotter()
    plotter.show()
    print("B")
    sys.exit(app.exec_())



# asyncio.run(scan())
