FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y && \
    apt-get install -y \
    git \
    apt-utils
ENV TZ=Asia/Tokyo

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:jonathonf/python-3.6 -y && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3.6 \
    python3.6-dev \
    python3.6-tk \
    tzdata \
    gcc \
    g++ \
    make \
    cmake \
    wget \
    bzip2 \
    libblas-dev \
    liblapack-dev \
    libfreetype6-dev \
    pkg-config \
    && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN wget http://ftp.tsukuba.wide.ad.jp/software/gcc/releases/gcc-7.3.0/gcc-7.3.0.tar.gz && \
    tar -zxvf gcc-7.3.0.tar.gz
RUN mkdir -p /gcc-7.3.0/build
WORKDIR /gcc-7.3.0
RUN ./contrib/download_prerequisites
WORKDIR /gcc-7.3.0/build
RUN ../configure --enable-languages=c,c++ --prefix=/usr/local --disable-bootstrap --disable-multilib && \
    make && make install

WORKDIR /
RUN rm /usr/bin/python3 && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.6 get-pip.py && \
    ln -s /usr/bin/python3.6 /usr/bin/python3 && \
    ln -s /usr/bin/pip3.6 /usr/bin/pip3.6

RUN git clone -b bugfix/gpu_feature_20190427_2 https://github.com/corryvrequan/qulacs.git
WORKDIR qulacs
RUN python3 setup_gpu.py install

WORKDIR /
RUN git clone https://github.com/quantumlib/Cirq.git
WORKDIR /Cirq
RUN apt-get update -y && \
    cat apt-system-requirements.txt dev_tools/conf/apt-list-dev-tools.txt | xargs apt-get install --yes
RUN python3 setup.py install

WORKDIR /
RUN git clone -b prototype https://github.com/qulacs/cirq-qulacs.git
WORKDIR /cirq-qulacs
RUN python3 setup.py install
RUN python3 setup.py test

WORKDIR /
