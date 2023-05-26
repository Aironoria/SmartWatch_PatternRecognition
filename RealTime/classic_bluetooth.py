#
# smartwatch_MAC = "B0:4A:6A:13:86:4B"
# import socket
#
# port = 3 # 3 is an arbitrary choice. However, it must match the port used by the client.
# backlog = 1
# size = 1024
# s = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
# s.bind((smartwatch_MAC,port))
# s.listen(backlog)
# try:
#     client, address = s.accept()
#     while 1:
#         data = client.recv(size)
#         if data:
#             print(data)
#             # client.send(data)
# except:
#     print("Closing socket")
#     client.close()
#     s.close()
import bluetooth

hostMACAddress = "B0:4A:6A:13:86:4B" # The MAC address of a Bluetooth adapter on the server. The server might have multiple Bluetooth adapters.
port = 0
# port = 0x1001
backlog = 1
size = 1024
print("start")
s = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
s.bind(("", port))
s.listen(backlog)
try:
    client, clientInfo = s.accept()
    print("accepted")
    while 1:
        data = client.recv(size)
        if data:
            print(data)
            client.send(data) # Echo back to client
except:
    print("Closing socket")
    client.close()
    s.close()


