#!/bin/bash
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
${HUGGINGFACE_TOKEN}=""
mkdir -p ~/.huggingface
echo -n ${HUGGINGFACE_TOKEN} > ~/.huggingface/token
wget -q https://github.com/ShivamShrirao/diffusers/raw/main/examples/dreambooth/train_dreambooth.py
wget -q https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py
pip install -qq git+https://github.com/ShivamShrirao/diffusers
pip install -q -U --pre triton
pip install -q accelerate transformers ftfy bitsandbytes==0.35.0 gradio natsort safetensors xformers
pip install git+https://github.com/facebookresearch/xformers@4c06c79#egg=xformers
pip install bitsandbytes-cuda117
pip install -U numpy