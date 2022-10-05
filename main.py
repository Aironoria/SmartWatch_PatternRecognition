import  os
import  subprocess

cmd = "ping baidu.com"
pwd = "cd /Users && pwd"

p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

line = p.stdout.readline().decode("utf-8").strip()
print(line)

while True:
    line = p.stdout.readline().decode("utf-8").strip()
    print(line)
    