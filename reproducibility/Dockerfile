# Start with an official Ubuntu 16.04 base image
FROM ubuntu:16.04

# Set environment variables to prevent prompts during installations
ENV DEBIAN_FRONTEND=noninteractive

# Update and install required system packages
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    zip \
    unzip \
    tar \
    && apt-get clean

# Download and install Anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh -O /tmp/anaconda.sh && \
    bash /tmp/anaconda.sh -b -p /anaconda && \
    rm /tmp/anaconda.sh

# Set Anaconda path
ENV PATH="/anaconda/bin:$PATH"

# Create a Conda environment and install required libraries
RUN conda create -n threatrace_env python=3.6.13 -y && \
    conda run -n threatrace_env conda install -y \
        pytorch==1.9.1 \
        torchvision==0.10.1 \
        torchaudio==0.9.1 \
        cudatoolkit=10.2 \
        psutil \
        -c pytorch && \
    conda run -n threatrace_env pip install \
        torch-cluster==1.5.9 \
        torch-scatter==2.0.9 \
        torch-sparse==0.6.12 \
        torch-spline-conv==1.2.1 \
        -f https://data.pyg.org/whl/torch-1.9.1+cu102.html && \
    conda run -n threatrace_env pip install torch-geometric==1.4.3 && \
    conda clean -a

# Correct the incompatibility issue in the torch-geometric library
RUN sed -i 's/from torch._six import container_abcs, string_classes, int_classes/import collections.abc as container_abcs\nstring_classes = str\nint_classes = int/' \
    /anaconda/envs/threatrace_env/lib/python3.6/site-packages/torch_geometric/data/dataloader.py

# Set the default environment when running the container
ENV CONDA_DEFAULT_ENV=threatrace_env
ENV PATH="/anaconda/envs/threatrace_env/bin:$PATH"

# Set working directory
WORKDIR /workspace

# Default command
CMD ["bash"]