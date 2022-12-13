import socket
from datetime import datetime

from realtimeplotter import RealTimePlotter

#home
ip_port = ('192.168.0.207',8081)

#lab
# ip_port=('172.24.195.54',8081)

#personal hotspot
# ip_port=('172.20.10.5',8081)

server = socket.socket()
server.bind(ip_port)
server.listen()
sock,addr = server.accept()
# data = ""
last_second =""
prefix=""
count =0
axis_num = 6
plotter = RealTimePlotter(axis_num,200)
while True:
    tmp_data = sock.recv(1024, socket.MSG_WAITALL)
    if not tmp_data:
        continue
    tmp_data = tmp_data.decode("utf8")
    # print(tmp_data)

    for item in tmp_data.split("\n"):
        if not prefix =="":
            item =prefix +item
            prefix=""

        if not item.endswith(";"):
            prefix = item
            continue
        print(datetime.now().strftime("%H:%M:%S.%f"),end=" ")
        item = item.split(";")[0]
        print(item)
        second =item.split(" ")[0].split(":")[2]
        data = item.split(" ")[1].split(",")
        if not second == last_second:
            print(str(count) +" hz")
            last_second = second
            count=0
        else:
            count+=1
        plotter.add_data(data[:axis_num])
# print('%s发送的内容：%s'%(addr[0],data))
sock.close()



def update_chart(data):
    #update data

    pass