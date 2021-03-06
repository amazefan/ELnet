FROM nvidia/cuda:8.0-cudnn5-devel

# Install curl and sudo
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
 && rm -rf /var/lib/apt/lists/*

# Use Tini as the init process with PID 1
RUN curl -Lso /tini https://github.com/krallin/tini/releases/download/v0.14.0/tini \
 && chmod +x /tini
ENTRYPOINT ["/tini", "--"]

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Install Git, bzip2, and X11 client
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    git \
    bzip2 \
    libx11-6 \
 && sudo rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.3.14-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh

# Create a Python 3.6 environment
RUN ~/miniconda/bin/conda install conda-build \
 && ~/miniconda/bin/conda create -y --name pytorch-py36 \
    python=3.6.0 numpy pyyaml scipy ipython mkl \
 && ~/miniconda/bin/conda clean -ya
ENV PATH=/root/miniconda/envs/pytorch-py36/bin:$PATH \
    CONDA_DEFAULT_ENV=pytorch-py36 \
    CONDA_PREFIX=/root/miniconda/envs/pytorch-py36

# CUDA 8.0-specific steps
RUN conda install -y --name pytorch-py36 -c soumith \
    magma-cuda80 \
 && conda clean -ya

# Install PyTorch and Torchvision
RUN conda install -y --name pytorch-py36 -c soumith \
    pytorch=0.2.0 torchvision=0.1.9 \
 && conda clean -ya

# Install HDF5 Python bindings
RUN conda install -y --name pytorch-py36 \
    h5py \
 && conda clean -ya
RUN pip install h5py-cache

# Install Torchnet, a high-level framework for PyTorch
RUN pip install git+https://github.com/pytorch/tnt.git@master

# Install Requests, a Python library for making HTTP requests
RUN conda install -y --name pytorch-py36 requests && conda clean -ya

# Install Graphviz
RUN conda install -y --name pytorch-py36 graphviz=2.38.0 \
 && conda clean -ya
RUN pip install graphviz

# Install OpenCV3 Python bindings
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module \
 && sudo rm -rf /var/lib/apt/lists/*
RUN conda install -y --name pytorch-py36 -c menpo opencv3 \
 && conda clean -ya

# Set the default command to python3
CMD ["python3"]
