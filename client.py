from __future__ import print_function
import requests
import json
import cv2

addr = 'http://localhost:5000'
test_url = addr + '/train'
data = {'instance_prompt' : 'abc',
        'class_prompt' : 'abc'}
test_files = {('images', open("/home/list99/workspace_dactt/Dreambooth_Stable_Diffusion-demo/image_0.png", "rb")),
              ('images', open("/home/list99/workspace_dactt/Dreambooth_Stable_Diffusion-demo/data/instance/3.png", "rb"))}
response = requests.post(test_url, files = test_files, data = data)
# encode image as jpeg
# send http request with image and receive response
response = requests.post(test_url)
# decode response
print(response.text)