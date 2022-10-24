import http.client
import json
import os.path


headers = { 'x-jwt-token': "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOjc0Njg0LCJpYXQiOjE2NjU0MTA4MzUsImV4cCI6MTY2ODAwMjgzNX0.jlsZUTUrZR8BpT5P4Fxnwa1FsphZ0vVXsvBdBuzIb60" }

# conn.request("GET", "/v1/api/144419/raw-data?category=training", headers=headers)
# conn.request("GET", "/v1/api/144419/raw-data/139501956/raw", headers=headers)
#
# res = conn.getresponse()
# data = res.read().decode("utf-8")
# data = json.loads(data)
# print(data.decode("utf-8"))

# config = {
#     'aj':{
#         "project_id":"144419",
#         "headers": { 'x-jwt-token': "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOjc0Njg0LCJpYXQiOjE2NjU0MTA4MzUsImV4cCI6MTY2ODAwMjgzNX0.jlsZUTUrZR8BpT5P4Fxnwa1FsphZ0vVXsvBdBuzIb60"
#
#         }
#     },
#     '144419': {
#         "project_id": "144568",
#         "headers": {
#             'x-jwt-token':          "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOjc0Njg0LCJpYXQiOjE2NjU0MTU5MzgsImV4cCI6MTY2ODAwNzkzOH0.lAMmJOZpePh7nXRdjmE6Hw8kM4Z41HgwlhcaIXmunt8"
#
#         }
#     }
#
# }

config = {
    "aj_data":"144419",
    "poh":"144568",

}

aj_data = "144419"
poh = "144568"
feng__pengyou = "144561"
gao ="144599"

def export_files(project_id, category="training"):
    conn.request("GET", f"/v1/api/{project_id}", headers=headers)
    res = conn.getresponse()
    data = json.loads(res.read().decode("utf-8"))
    save_dir = "edgeimpulse_12/" + data['project']['name']
    # project_id = str(config[project_name]["project_id"])
    # headers = config[project_name]["headers"]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    conn.request("GET", "/v1/api/"+ project_id +"/raw-data?category=" + category, headers=headers)
    # data = conn.getresponse().read().decode("utf-8")
    file_list = json.loads(conn.getresponse().read().decode("utf-8"))['samples']
    for idx, file in enumerate(file_list):
        conn.request("GET", "/v1/api/"+ project_id +"/raw-data/" + str(file['id']) + "/raw", headers=headers)
        data = json.loads(conn.getresponse().read().decode("utf-8"))
        file_name = file['filename'].split(".")[-1]
        file_name =file_name.replace("F","L")
        label = file['label']
        b = json.dumps(data)
        f2 = open(save_dir + "/" + label + "." + file_name + ".json", 'w')
        f2.write(b)
        f2.close()
        print(str(idx + 1) + "/" + str(len(file_list)) + " Saved.")

conn = http.client.HTTPSConnection("studio.edgeimpulse.com")
# export_files(aj_data)
# export_files(poh)
export_files(feng__pengyou)
export_files(gao)