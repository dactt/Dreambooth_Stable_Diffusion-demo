from __future__ import print_function
import requests
import json
import cv2

addr = 'http://localhost:5000'
test_url = addr + '/test'

# encode image as jpeg
# send http request with image and receive response
response = requests.post(test_url)
# decode response
print(response.text)