import socket

from realtimeplotter import RealTimePlotter

# ip_port = ('192.168.0.155',8081)
ip_port=('172.24.195.52',8081)
server = socket.socket()
server.bind(ip_port)
server.listen()
sock,addr = server.accept()
# data = ""
last_second =""
count =0
axis_num = 3
plotter = RealTimePlotter(3)
while True:
    tmp_data = sock.recv(1024)
    if not tmp_data:
        continue
    tmp_data = tmp_data.decode("utf8")
    print(tmp_data, end="")
    if len(tmp_data.split(" ")) <2:
        continue
    time =tmp_data.split(" ")[0]
    if len(time.split(":")) < 4:
        continue
    data = tmp_data.split(" ")[1]
    if len(data.split(",")) < axis_num:
        continue

    data = data.split(",")
    second = time.split(":")[2]
    if not second == last_second:
        print(str(count) +" hz")
        last_second = second
        count=0
    else:
        count+=1
    # plotter.add_data(data[:3])
    # print(tmp_data)
# print('%s发送的内容：%s'%(addr[0],data))
sock.close()



def update_chart(data):
    #update data

    pass