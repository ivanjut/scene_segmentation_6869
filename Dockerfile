# ubuntu:18.04
FROM nvidia/cuda:10.2-base-ubuntu18.04
LABEL maintainer "Eric Chen - ericrc@mit.edu"
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    make \
    cmake \
    curl \
    gcc \
    git \
    tmux \
    htop \
    nano \
    python3.7 \
    python3-pip \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libosmesa6-dev \
    xvfb \
    ffmpeg \
    curl \
    patchelf \
    libglfw3 \
    libglfw3-dev \
    zlib1g \
    zlib1g-dev \
    swig \
    wget \
    build-essential \
    zlib1g-dev \
    libsdl2-dev \
    libjpeg-dev \
    nasm \
    tar \
    libbz2-dev \
    libgtk2.0-dev \
    libfluidsynth-dev \
    libgme-dev \
    libopenal-dev \
    libboost-all-dev \
    timidity \
    libwildmidi-dev \
    unzip \
    lsof \
    libjpeg-turbo8-dev \
    xorg-dev \
    libx11-dev \
    libxcursor-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxi-dev \
    libxxf86vm-dev \
    mesa-common-dev

ENV PYTHONPATH /scene_seg
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt && rm /tmp/requirements.txt

