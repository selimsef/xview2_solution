FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# prep apt-get and cudnn
RUN apt-get update && apt-get install -y --no-install-recommends \
	    apt-utils  && \
    rm -rf /var/lib/apt/lists/*

# install requirements
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    apt-utils \
    bc \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libgdal-dev \
    libssl-dev \
    libffi-dev \
    libncurses-dev \
    libgl1 \
    jq \
    nfs-common \
    parallel \
    python-dev \
    python-pip \
    python-wheel \
    python-setuptools \
    unzip \
    vim \
	tmux \
    wget \
    build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/bin:$PATH

# install anaconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# use conda-forge instead of default channel
RUN conda update conda && \
    #conda config --remove channels defaults && \
    conda config --add channels conda-forge

# set up conda environment and add to $PATH
RUN conda create -n cresi python=3.6 \
                    && echo "source activate cresi" > ~/.bashrc
ENV PATH /opt/conda/envs/cresi/bin:$PATH

RUN mkdir -p /root/.torch/models

RUN pip install torch==1.1.0 -f https://download.pytorch.org/whl/torch_stable
RUN pip install torchvision==0.4.0 --no-deps
RUN pip install albumentations==0.4.1
RUN pip install pretrainedmodels

RUN pip install tensorboardX \
	&& pip install torchsummary \
	&& pip install utm \
	&& pip install numba

RUN pip uninstall apex
RUN git clone https://github.com/NVIDIA/apex
RUN sed -i 's/check_cuda_torch_binary_vs_bare_metal(torch.utils.cpp_extension.CUDA_HOME)/pass/g' apex/setup.py
RUN  pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"  /apex
RUN conda install -n cresi \
     	      awscli \
              affine \
              pyhamcrest \
              cython \
              fiona \
              h5py \
              ncurses \
              jupyter \
              jupyterlab \
              ipykernel \
              matplotlib \
	          ncurses \
              numpy \
			  statsmodels \
              pandas \
              pillow \
              pip \
              scipy \
              scikit-image \
              scikit-learn \
              shapely \
              rtree \
              testpath \
              tqdm \
              pandas \
			  opencv \
	&& conda clean -p \
	&& conda clean -t \
	&& conda clean --yes --all
ENV LD_LIBRARY_PATH /miniconda/lib:${LD_LIBRARY_PATH}
RUN apt update
RUN pip install efficientnet_pytorch
RUN pip install Pillow==6.1
# add a jupyter kernel for the conda environment in case it's wanted
RUN source activate cresi && python -m ipykernel.kernelspec

# open ports for jupyterlab and tensorboard
EXPOSE 8888 6006

WORKDIR /work

COPY . /work/

RUN chmod +x run.sh

ENTRYPOINT ["./run.sh"]
