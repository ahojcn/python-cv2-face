import cv2

from aip import AipFace

import utils

client = AipFace(utils.BAIDU_AI_APP_ID, utils.BAIDU_AI_API_KEY, utils.BAIDU_AI_SECRET_KEY)
image_type = "BASE64"
group_id = "ahojcn"
user_id = "fancheng"
options = {"user_info": "范程"}

image = cv2.imread("./img/fancheng.jpg")
image_base64 = utils.image_to_base64(image)
# print(client.addUser(image_base64, image_type, group_id, user_id, options))

import requests, json

# host = f'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={utils.BAIDU_AI_API_KEY}&client_secret={utils.BAIDU_AI_SECRET_KEY}'
# response = requests.get(host)
# if response:
#     print(response.json())
access_token = '24.e4265a7b77a8cadbdb194334fa9b5cbf.2592000.1611411456.282335-18145377'

request_url = 'https://aip.baidubce.com/rest/2.0/face/v3/faceset/user/add'
data = {
    "image": image_base64,
    "image_type": "BASE64",
    "group_id": "ahojcn",
    "user_id": "fancheng",
    "user_info": "范程"
}
params = json.dumps(data)
request_url = request_url + "?access_token=" + access_token
headers = {'content-type': 'application/json'}
response = requests.post(request_url, data=params, headers=headers)
if response:
    print(response.json())
