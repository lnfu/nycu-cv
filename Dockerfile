FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Set noninteractive mode to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install required packages
RUN apt-get update && apt-get install -y \
    sudo \
    python3.12 \
    python3-pip \
    python-is-python3 \
    vim \
    git \
    make \
    ffmpeg \
    locales \
    speech-dispatcher \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python3.12 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

RUN pip install --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple --break-system-packages \
    torch==2.6.0+cu126 \
    torchaudio==2.6.0+cu126 \
    torchvision==0.21.0+cu126

RUN pip install uv --break-system-packages

RUN echo "ubuntu:ubuntu" | chpasswd

RUN usermod -aG sudo ubuntu
