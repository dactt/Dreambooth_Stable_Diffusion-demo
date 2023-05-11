import time
import os
import shutil
import json
import subprocess
from natsort import natsorted
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from flask import Flask, request, jsonify

app = Flask(__name__)
time_start = time.time()

HUGGINGFACE_TOKEN = ""
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
BRANCH = "fp16"
OUTPUT_DIR = os.getcwd() + "stable_diffusion_weights/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route('/train', methods=['POST'])
def train():
    # Get promt from request
    instance_prompt = request.form['instance_prompt']
    class_prompt = request.form['class_prompt']

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



instance_prompt = "photo of zwx's face" 
class_prompt =  "photo of a man face" 
training_steps = 800 
learning_rate = 1e-6 

fp16 = True 
complie_xformers = False 


OUTPUT_DIR = "stable_diffusion_weights/output"
OUTPUT_DIR = "../convert-Dreambooth_Stable_Diffusion/" + OUTPUT_DIR

print(f"[*] Weights will be saved at {OUTPUT_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# You can also add multiple concepts here. Try tweaking `--max_train_steps` accordingly.

concepts_list = [
    {
        "instance_prompt":      instance_prompt,
        "class_prompt":         class_prompt,
        "instance_data_dir":    "../convert-Dreambooth_Stable_Diffusion/data/instance",
        "class_data_dir":       "../convert-Dreambooth_Stable_Diffusion/data/class"
    },
]

for c in concepts_list:
    os.makedirs(c["instance_data_dir"], exist_ok=True)

with open("concepts_list.json", "w") as f:
    json.dump(concepts_list, f, indent=4)




for c in concepts_list:
    print(f"Uploading instance images for `{c['instance_prompt']}`")
    uploaded  = os.listdir(c['instance_data_dir'])
    # for filename in uploaded.keys():
    #     dst_path = os.path.join(c['instance_data_dir'], filename)
    #     shutil.move(filename, dst_path)

# # huggingface token
os.makedirs('~/.huggingface',exist_ok=True)
subprocess.run(f'echo -n "{HUGGINGFACE_TOKEN}" > ~/.huggingface/token',shell=True)

# ############## Edit this section to customize parameters

subprocess.run(f'python3 train_dreambooth.py \
  --pretrained_model_name_or_path={MODEL_NAME} \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --output_dir={OUTPUT_DIR} \
  --revision={BRANCH} \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --seed=1337 \
  --resolution=512 \
  --train_batch_size=1 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --learning_rate={learning_rate} \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=50 \
  --sample_batch_size=4 \
  --max_train_steps={training_steps} \
  --save_interval=10000 \
  --save_sample_prompt="{instance_prompt}" \
  --concepts_list="concepts_list.json"',shell=True)

# ########################################

# # Reduce the `--save_interval` to lower than `--max_train_steps` to save weights from intermediate steps.
# # `--save_sample_prompt` can be same as `--instance_prompt` to generate intermediate samples (saved along with weights in samples directory).


weightdirs = natsorted(glob(OUTPUT_DIR + os.sep + "*"))
if len(weightdirs) == 0:
  raise KeyboardInterrupt("No training weights directory found")
WEIGHTS_DIR = weightdirs[-1]


ckpt_path = WEIGHTS_DIR + "/model.ckpt"

half_arg = ""
if fp16:
    half_arg = "--half"
subprocess.run(f"python convert_diffusers_to_original_stable_diffusion.py --model_path {WEIGHTS_DIR}  --checkpoint_path {ckpt_path} {half_arg}",shell=True)
print(f"[*] Converted ckpt saved at {ckpt_path}")

print(f"[*] WEIGHTS_DIR={WEIGHTS_DIR}")
minutes = (time.time()-time_start)/60
print("Dreambooth completed successfully. It took %1.1f minutes."%minutes)


weights_folder = OUTPUT_DIR
folders = sorted([f for f in os.listdir(weights_folder) if f != "0"], key=lambda x: int(x))

row = len(folders)
col = len(os.listdir(os.path.join(weights_folder, folders[0], "samples")))
scale = 4
fig, axes = plt.subplots(row, col, figsize=(col*scale, row*scale), gridspec_kw={'hspace': 0, 'wspace': 0})

for i, folder in enumerate(folders):
    folder_path = os.path.join(weights_folder, folder)
    image_folder = os.path.join(folder_path, "samples")
    images = [f for f in os.listdir(image_folder)]
    for j, image in enumerate(images):
        if row == 1:
            currAxes = axes[j]
        else:
            currAxes = axes[i, j]
        if i == 0:
            currAxes.set_title(f"Image {j}")
        if j == 0:
            currAxes.text(-0.1, 0.5, folder, rotation=0, va='center', ha='center', transform=currAxes.transAxes)
        image_path = os.path.join(image_folder, image)
        img = mpimg.imread(image_path)
        currAxes.imshow(img, cmap='gray')
        currAxes.axis('off')

plt.tight_layout()
plt.savefig('grid.png', dpi=72)
