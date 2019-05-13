FROM ubuntu:16.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:jonathonf/python-3.6 -y && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3.6 \
    python3.6-dev \
    python3.6-tk \
    git \
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
    libpng12-dev \
    && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update -y && \
    apt-get update -y && \
    apt-get install g++-7 -y

WORKDIR /
RUN rm /usr/bin/python3 && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.6 get-pip.py && \
    ln -s /usr/bin/python3.6 /usr/bin/python3 && \
    ln -s /usr/bin/pip3.6 /usr/bin/pip3.6

RUN pip3 install \
    numpy \
    matplotlib \
    setuptools

RUN git clone https://github.com/qulacs/qulacs.git
WORKDIR qulacs
RUN python3 setup.py install

WORKDIR /
RUN git clone https://github.com/quantumlib/Cirq.git
WORKDIR /Cirq
RUN apt-get update -y && \
    cat apt-system-requirements.txt dev_tools/conf/apt-list-dev-tools.txt | xargs apt-get install --yes
RUN python3 setup.py install

RUN git clone https://github.com/qulacs/cirq-qulacs.git
WORKDIR /cirq-qulacs
RUN python3 setup.py install
RUN python3 setup.py test

WORKDIR /
RUN pip3 install qiskit==0.6.0
RUN git clone https://github.com/Qiskit/openqasm.git 
