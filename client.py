from __future__ import print_function
import requests
import json
import cv2

def test_train():
        addr = 'http://localhost:5000'
        test_url = addr + '/train'
        # instance promt: describe the instance images uploaded to server
        # class promt: describe the same images as instance images, that may be in the dataset of stable-diffusion
        data = {
                'instance_prompt' : 'a photo of XYZ man face',
                'class_prompt' : 'a photo of man face'
                }
        # the images uploaded to server
        test_files = {
                ('images', open("/workspace/Dreambooth_Stable_Diffusion-demo/test_img/1.png", "rb")),
                ('images', open("/workspace/Dreambooth_Stable_Diffusion-demo/test_img/2.png", "rb")),
                ('images', open("/workspace/Dreambooth_Stable_Diffusion-demo/test_img/3.png", "rb")),
                ('images', open("/workspace/Dreambooth_Stable_Diffusion-demo/test_img/4.png", "rb")),
            }
        response = requests.post(test_url, files = test_files, data = data)
        print(response.text)

def test_inference():
        addr = 'http://localhost:5000'
        test_url = addr + '/inference'
        data = {'prompt' : 'oil painting of XYZ man face in naruto style, only face',
                'negative_prompt' : '',
                'num_samples' : 4,
                'guidance_scale' : 7.5,
                'num_inference_steps' : 30,
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