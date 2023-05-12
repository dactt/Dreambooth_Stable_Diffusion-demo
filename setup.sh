#!/bin/bash
#wget -q https://github.com/ShivamShrirao/diffusers/raw/main/examples/dreambooth/train_dreambooth.py
#wget -q https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py
conda init
conda create -n dreamboothSD -yq
source /opt/conda/bin/activate dreamboothSD
pip install -qq git+https://github.com/ShivamShrirao/diffusers
pip install -q -U --pre triton
pip install -q accelerate transformers ftfy bitsandbytes==0.35.0 gradio natsort safetensors xformers flask
pip install git+https://github.com/facebookresearch/xformers@4c06c79#egg=xformers
pip install bitsandbytes-cuda117
pip install -U numpy