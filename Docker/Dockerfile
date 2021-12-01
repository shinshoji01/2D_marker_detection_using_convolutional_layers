FROM nvidia/cuda:11.1.1-runtime-ubuntu20.04

COPY . /build_context
WORKDIR /opt

ENV TZ Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
	    libsndfile-dev \
	    python3 \
	    python3-pip \
	    sudo \
	    wget \
	    vim

# nodejs
RUN sudo apt-get install -y nodejs npm && \
	    sudo npm install n -g && \
	    sudo n stable && \
	    sudo apt purge -y nodejs npm

# python packages
RUN pip3 install --upgrade pip && \
	    pip3 install -r /build_context/requirements.txt

RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /work

CMD ["bash"]
