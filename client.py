from __future__ import print_function
import requests
import json
import cv2

def test_train():
        addr = 'http://localhost:5000'
        test_url = addr + '/train'
        data = {'instance_prompt' : 'abc',
                'class_prompt' : 'abc'}
        test_files = {('images', open("/home/list99/workspace_dactt/Dreambooth_Stable_Diffusion-demo/image_0.png", "rb")),
                ('images', open("/home/list99/workspace_dactt/Dreambooth_Stable_Diffusion-demo/data/instance/3.png", "rb"))}
        response = requests.post(test_url, files = test_files, data = data)
        print(response.text)

def test_inference():
        addr = 'http://localhost:5000'
        test_url = addr + '/inference'
        data = {'prompt' : 'abc',
                'negative_prompt' : 'a',
                'num_samples' : 4,
                'guidance_scale' : 7.5,
                'num_inference_steps' : 1,
                'height' : 512,
                'width' : 512,
                'seed' : 1,
                }
        response = requests.post(test_url, data = data)
        if response.status_code == 200:
            with open('images_download.zip', 'wb') as file:
                file.write(response.content)
            print('File downloaded successfully')
        else:
            print('Falied to download')

if __name__== "__main__":
        test_inference()