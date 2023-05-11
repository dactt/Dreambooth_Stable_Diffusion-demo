import io
import json
import os
import subprocess
import time
import zipfile
from glob import glob

import PIL
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from flask import Flask, Response, jsonify, request, send_file
from natsort import natsorted
from torch import autocast

app = Flask(__name__)
time_start = time.time()

HUGGINGFACE_TOKEN = ""
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
BRANCH = "fp16"
OUTPUT_DIR = os.getcwd() + "stable_diffusion_weights/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[*] Weights will be saved at {OUTPUT_DIR}")

TRAINING_STEPS = 800 
LR = 1e-6
BATCH_SIZE = 1
SEED = 0
RESOLUTION = 512

CLASS_IMAGE = 50    # num of image that using class_prompt to generate

FP16 = True # convert trained model to new model using fp16 or fp32

@app.route('/train', methods=['POST'])
def train():
    # Get promt from request
    instance_prompt = request.form['instance_prompt']
    class_prompt = request.form['class_prompt']

    #instance_prompt = "photo of zwx's face" 
    #class_prompt =  "photo of a man face" 

    # Get instance image from request
    images = request.files.getlist('images')

    instance_image_dir = os.path.join(os.getcwd(), '/data/instance')
    class_image_dir = os.path.join(os.getcwd(), '/data/class')
    
    os.makedirs(instance_image_dir,exist_ok=True)
    print(f"Uploading instance images for promt : `{instance_prompt}`")
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(instance_image_dir, f'image_{i}.jpg')
        image.save(image_path)
        image_paths.append(image_path)

    concepts_list = [
        {
            "instance_prompt":      instance_prompt,
            "class_prompt":         class_prompt,
            "instance_data_dir":    instance_image_dir,
            "class_data_dir":       class_image_dir
        },
    ]

    with open("concepts_list.json", "w") as f:
        json.dump(concepts_list, f, indent=4)

    print('-----------------Start training-----------------')
    subprocess.run(f'CUDA_VISIBLE_DEVICES=0 python3 train_dreambooth.py \
        --pretrained_model_name_or_path={MODEL_NAME} \
        --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
        --output_dir={OUTPUT_DIR} \
        --revision={BRANCH} \
        --with_prior_preservation \
        --prior_loss_weight=1.0 \
        --seed={SEED} \
        --resolution={RESOLUTION} \
        --train_batch_size={BATCH_SIZE} \
        --train_text_encoder \
        --mixed_precision="fp16" \
        --use_8bit_adam \
        --gradient_accumulation_steps=1 \
        --learning_rate={LR} \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --num_class_images={CLASS_IMAGE} \
        --sample_batch_size=4 \
        --max_train_steps={TRAINING_STEPS} \
        --save_interval=10000 \
        --save_sample_prompt="{instance_prompt}" \
        --concepts_list="concepts_list.json"'
        ,shell=True)
    print('-----------------Finish training-----------------')

    weightdirs = natsorted(glob(OUTPUT_DIR + "/" + "*"))
    if len(weightdirs) == 0:
        raise KeyboardInterrupt("No training weights directory found")
    WEIGHTS_DIR = weightdirs[-1]
    ckpt_path = WEIGHTS_DIR + "/model.ckpt"

    if FP16:
        half_arg = "--half"
        subprocess.run(f"python3 convert_diffusers_to_original_stable_diffusion.py \
                       --model_path {WEIGHTS_DIR}  \
                       --checkpoint_path {ckpt_path} {half_arg}"
                       ,shell=True)
    print(f"[*] Converted ckpt saved at {ckpt_path}")
    print(f"[*] WEIGHTS_DIR={WEIGHTS_DIR}")
    print("Dreambooth Train Completed. It took %1.1f minutes." % (time.time()-time_start)/60)

    response = {'ckpt_path' : ckpt_path}

    return Response(response=response, status=200)
    return ckpt_path

@app.route('/inference', methods=['POST'])
def inference():

    prompt = request.form['prompt']
    negative_prompt = request.form['negative_prompt']
    num_samples = request.form['num_samples']
    guidance_scale = request.form['guidance_scale']
    num_inference_steps = request.form['num_inference_steps']
    height = request.form['height']
    width = request.form['width']
    seed = request.form['seed']

    model_path = ''
    if 'pipe' not in locals():
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16).to("cuda")
        g_cuda = None

    g_cuda = torch.Generator(device='cuda')

    g_cuda.manual_seed(seed)

    with autocast('cuda'), torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=g_cuda
        ).images

    zip_file_name = 'images.zip'
    with zipfile.ZipFile(zip_file_name, mode='w') as zip_file:
        for i, image in enumerate(images):
            tmp_file = f'image_{i}.png'
            image.save(tmp_file)
            zip_file.write(tmp_file)
    return send_file(zip_file_name)
@app.route('/test', methods=['POST'])
def test():
    import json
    images = []
    for i in range (4):
        img = PIL.Image.open('/home/list99/workspace_dactt/convert-Dreambooth_Stable_Diffusion/data/instance/1.png')
        images.append(img)
    zip_data = 'images.zip'
    with zipfile.ZipFile(zip_data, mode='w') as zip_file:
        for i, image in enumerate(images):
            tmp_file = f'image_{i}.png'
            image.save(tmp_file)
            zip_file.write(tmp_file)
    response = {'ckpt_path' : "ok nho"}
    return Response(response=json.dumps(response), status=200)

if __name__=='__main__':
    app.run(host="0.0.0.0", port=5000)
    #test()