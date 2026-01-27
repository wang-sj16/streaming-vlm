conda create -n streamingvlm-sft python=3.11 -y
conda activate streamingvlm-sft
pip install torch==2.7.1 torchvision torchaudio
pip install "transformers<=4.51.3" accelerate deepspeed peft opencv-python decord datasets tensorboard gradio pillow-heif gpustat timm sentencepiece openai av==12.0.0 qwen_vl_utils liger_kernel numpy==1.24.4
pip install flash-attn==2.8.0.post2  --no-build-isolation --force-reinstall
pip install deepspeed==0.17.1
pip install qwen_vl_utils==0.0.11 wandb
pip install -e streaming_vlm/livecc_utils/

# install flash-attn
# find your version at https://github.com/Dao-AILab/flash-attention/releases
# example:
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl